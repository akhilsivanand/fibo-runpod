"""ComfyUI gateway with preparation pages and optional authentication."""
import asyncio
import base64
from dataclasses import dataclass
import hashlib
import hmac
import html
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import time
from urllib.parse import urlsplit, quote, parse_qs

import aiohttp
from aiohttp import web

HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'host', 'authorization'}
UPSTREAM_SESSION = web.AppKey('upstream', aiohttp.ClientSession)
SESSION_COOKIE = '__Host-fibo-session'
SESSION_TTL = 12 * 60 * 60


@dataclass(frozen=True)
class GatewayAuth:
    mechanism: str
    secret: str
    port: int


def validate_secret(secret, name):
    if not isinstance(secret, str) or len(secret) < 12 or len(secret) > 4096:
        raise ValueError(f'{name} must contain between 12 and 4096 characters.')
    if 'RUNPOD_SECRET_' in secret.upper():
        raise ValueError(f'{name} contains an unresolved Runpod Secret placeholder; select an existing Runpod Secret or remove this override.')
    if any(ord(character) < 32 or ord(character) == 127 for character in secret):
        raise ValueError(f'{name} must not contain control characters.')
    try:
        secret.encode('utf-8')
    except UnicodeError:
        raise ValueError(f'{name} must contain valid UTF-8 text.') from None
    return secret


def auth_from_environment(mode='pod'):
    """Protect ComfyUI on 8188 without taking over Jupyter's port 8888."""
    if mode == 'serverless':
        return GatewayAuth('none', '', 8188)
    password = os.environ.get('FIBO_PASSWORD', '')
    if password:
        return GatewayAuth('basic', validate_secret(password, 'FIBO_PASSWORD'), 8188)
    token = os.environ.get('JUPYTER_PASSWORD', '')
    if token:
        return GatewayAuth('token', validate_secret(token, 'JUPYTER_PASSWORD'), 8188)
    raise ValueError('ComfyUI authentication is missing. Enable Jupyter so Runpod supplies a JUPYTER_PASSWORD token, or set a unique FIBO_PASSWORD of at least 12 characters for port 8188.')


def session_cookie(secret, now=None):
    issued = int(time.time() if now is None else now)
    payload = f'{issued}.{issued + SESSION_TTL}.{secrets.token_urlsafe(18)}'
    signature = hmac.new(secret.encode('utf-8'), payload.encode('ascii'), hashlib.sha256).hexdigest()
    return payload + '.' + signature


def session_authorized(value, secret, now=None):
    if not isinstance(value, str) or len(value) > 256:
        return False
    try:
        issued, expires, nonce, signature = value.split('.')
        if not re.fullmatch(r'[0-9]{1,12}', issued) or not re.fullmatch(r'[0-9]{1,12}', expires):
            return False
        if not re.fullmatch(r'[A-Za-z0-9_-]{24}', nonce) or not re.fullmatch(r'[a-f0-9]{64}', signature):
            return False
        current = int(time.time() if now is None else now)
        if not (int(issued) <= current < int(expires) and int(expires) - int(issued) == SESSION_TTL):
            return False
        payload = '.'.join((issued, expires, nonce))
        expected = hmac.new(secret.encode('utf-8'), payload.encode('ascii'), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)
    except (ValueError, UnicodeError):
        return False


def normalize_origin(value):
    """Parse a single HTTP Origin without trusting forwarded request headers."""
    if not isinstance(value, str) or not value or len(value) > 2048:
        return None
    if ('?' in value or '#' in value
            or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)):
        return None
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment):
            return None
        host = parsed.hostname.encode('idna').decode('ascii').lower()
        if '%' in host or parsed.netloc.endswith(':'):
            return None
        try:
            host = ipaddress.ip_address(host).compressed.lower()
        except ValueError:
            if len(host) > 253 or not all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', part)
                                         for part in host.split('.')):
                return None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == 'https' else 80
        if port < 1:
            return None
        return parsed.scheme, host, port
    except (ValueError, UnicodeError):
        return None


def public_origin_from_environment(port=8188):
    configured = os.environ.get('FIBO_PUBLIC_ORIGIN')
    if configured is not None:
        origin = normalize_origin(configured)
        if origin is None:
            raise ValueError('FIBO_PUBLIC_ORIGIN must be an HTTP(S) origin without a path, query, or credentials.')
        return origin
    pod_id = os.environ.get('RUNPOD_POD_ID')
    if pod_id is not None:
        if not re.fullmatch(r'[a-z0-9]{8,58}', pod_id):
            raise ValueError('RUNPOD_POD_ID is not a valid Pod identifier.')
        return normalize_origin('https://' + pod_id + f'-{port}.proxy.runpod.net')
    return None


def origin_allowed(origin, request_host, public_origin):
    if origin is None:
        return True  # Non-browser API clients need not send a browser Origin.
    normalized = normalize_origin(origin)
    if normalized is None:
        return False
    if public_origin is not None:
        return normalized == public_origin
    # Direct/local use, or a proxy preserving the public Host. TLS may terminate
    # before this gateway, so the request's internal HTTP scheme is not public.
    expected = normalize_origin(normalized[0] + '://' + request_host)
    return normalized == expected


def authorized(header, password):
    try:
        scheme, encoded = header.split(' ', 1)
        if scheme.lower() != 'basic':
            return False
        decoded = base64.b64decode(encoded, validate=True).decode('utf-8')
        username, secret = decoded.split(':', 1)
        return hmac.compare_digest(username.encode('utf-8'), b'fibo') and hmac.compare_digest(secret.encode('utf-8'), password.encode('utf-8'))
    except (ValueError, UnicodeError):
        return False


def page(state, workflow_config):
    ready = state.get('ready') is True
    cards = []
    if ready:
        entries = list(workflow_config['presets'].values()) + list(workflow_config.get('ui_only', {}).values())
        for entry in entries:
            name = Path(entry['ui']).stem
            href = '/?template=' + quote(name) + '&source=fibo'
            cards.append(f'<a class="card" href="{html.escape(href)}">{html.escape(entry["title"])}<span>Open workflow →</span></a>')
    phases = {
        'checking_gpu': ('Starting your studio', 'Preparing your workspace.'),
        'waiting_for_models': ('Preparing models', 'Model setup is already running.'),
        'checking_storage': ('Preparing models', 'Checking available space for your models.'),
        'downloading': ('Downloading models', 'Your studio is downloading the models it needs.'),
        'verifying': ('Checking models', 'Checking your model files before opening ComfyUI.'),
        'verified': ('Preparing models', 'Finishing model setup.'),
        'recovering': ('Retrying a download', 'A model file needs another download. Setup will continue automatically.'),
        'models_ready': ('Opening ComfyUI', 'Your models are ready. Preparing your workflows.'),
        'starting_comfyui': ('Opening ComfyUI', 'Your models are ready. Preparing your workflows.'),
    }
    phase_key = state.get('phase')
    phase, detail = phases.get(phase_key if isinstance(phase_key, str) else '',
                              ('Starting your studio', 'Preparing your workspace.'))
    if ready:
        phase, detail = 'Ready', 'Choose a workflow to open it in ComfyUI.'

    def byte_count(value):
        if isinstance(value, bool):
            return 0
        try:
            return min(max(int(value), 0), 2**63 - 1)
        except (TypeError, ValueError, OverflowError):
            return 0

    total = byte_count(state.get('bytes_total'))
    complete = min(byte_count(state.get('bytes_complete')), total)
    progress = ''
    if not ready:
        if total:
            percent = complete * 100 // total
            progress = f'<div class="progress"><div class="numbers"><span>{complete / 1e9:.1f} GB of {total / 1e9:.1f} GB</span><span>{percent}%</span></div><progress aria-label="Model preparation" value="{percent}" max="100">{percent}%</progress></div>'
        else:
            progress = '<div class="progress"><progress aria-label="Preparing your studio"></progress></div>'
    refresh = '' if ready else '<meta http-equiv="refresh" content="5">'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>Fibo Edit 2 Studio</title>
<style>body{{margin:0;background:#101815;color:#eef4ed;font:17px system-ui,sans-serif}}main{{max-width:900px;margin:10vh auto;padding:28px}}.label{{color:#a4cdb0;letter-spacing:.12em;font-size:13px}}h1{{font-size:clamp(36px,6vw,62px);font-weight:550;letter-spacing:-.04em;margin:16px 0}}p{{line-height:1.6;color:#c3cec5}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:36px}}.card{{padding:25px;background:#1b2922;border:1px solid #344a3c;border-radius:12px;color:#fff;text-decoration:none}}.card:hover{{border-color:#a4cdb0}}.card span{{display:block;color:#a4cdb0;margin-top:22px;font-size:14px}}.editor{{display:inline-block;color:#b7d7bf;margin-top:28px}}.status{{color:#a4cdb0}}.progress{{max-width:580px;margin:28px 0}}.numbers{{display:flex;justify-content:space-between;gap:16px;color:#eef4ed;font-variant-numeric:tabular-nums;margin-bottom:10px}}progress{{display:block;width:100%;height:12px;border:0;border-radius:8px;overflow:hidden;background:#26392e;accent-color:#a4cdb0}}progress::-webkit-progress-bar{{background:#26392e}}progress::-webkit-progress-value{{background:#a4cdb0}}progress::-moz-progress-bar{{background:#a4cdb0}}</style>
<main><div class="label">FIBO EDIT 2</div><h1>{'Your next edit starts here.' if ready else 'Preparing your studio.'}</h1><div class="status">{html.escape(phase)}</div><p>{html.escape(detail)}</p>{progress}<div class="grid">{''.join(cards)}</div>
{'<a class="editor" href="/?editor=1">Continue in your ComfyUI workspace →</a>' if ready else '<p>This page refreshes automatically. The first start downloads the models; later starts reuse them.</p>'}</main></html>'''


def create_app(state, root, password=None, mode='pod', upstream='http://127.0.0.1:8189', auth=None):
    config = json.loads((Path(root) / 'config/workflows.json').read_text())
    if auth is None:
        auth = (GatewayAuth('basic', validate_secret(password, 'FIBO_PASSWORD'), 8188)
                if mode == 'pod' and password is not None else auth_from_environment(mode))
    public_origin = public_origin_from_environment(auth.port) if mode == 'pod' else None
    app = web.Application(client_max_size=50 * 1024**2)

    async def session_context(application):
        application[UPSTREAM_SESSION] = aiohttp.ClientSession(auto_decompress=False,
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=15))
        yield
        await application[UPSTREAM_SESSION].close()

    app.cleanup_ctx.append(session_context)

    def unauthorized(request):
        headers = {'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'}
        if auth.mechanism == 'basic':
            headers['WWW-Authenticate'] = 'Basic realm="Fibo Edit 2 Studio", charset="UTF-8"'
            detail = 'Sign in with username fibo and your FIBO_PASSWORD.'
        else:
            detail = 'Sign in using the configured gateway token.'
            if request.method == 'GET':
                headers['Content-Security-Policy'] = "default-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
                return web.Response(status=401, content_type='text/html', headers=headers, text='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in to Fibo ComfyUI</title><h1>Fibo ComfyUI</h1>
<p>Enter your Pod's Jupyter token to open ComfyUI.</p>
<form method="post" action="/fibo/login"><label for="fibo-token">Jupyter token</label>
<input id="fibo-token" name="token" type="password" required maxlength="4096" autocomplete="current-password">
<button type="submit">Open ComfyUI</button></form></html>''')
        return web.Response(status=401, text=detail, headers=headers)

    def with_session(response):
        response.set_cookie(SESSION_COOKIE, session_cookie(auth.secret), max_age=SESSION_TTL,
                            path='/', secure=True, httponly=True, samesite='Strict')
        return response

    async def dispatch(request):
        if request.path in ('/healthz', '/readyz', '/ping'):
            if request.method != 'GET':
                raise web.HTTPMethodNotAllowed(request.method, ['GET'])
            ready = bool(state.get('ready'))
            if request.path == '/ping':
                return web.Response(status=200 if ready else 204)
            code = 200 if request.path == '/healthz' or ready else 503
            return web.json_response({'ready': ready, 'phase': state.get('phase', 'starting')}, status=code)
        if mode != 'pod':
            raise web.HTTPNotFound()
        native_token = auth.mechanism == 'token'
        if native_token and request.path == '/fibo/login':
            if request.method != 'POST':
                raise web.HTTPMethodNotAllowed(request.method, ['POST'])
            if not origin_allowed(request.headers.get('Origin'), request.host, public_origin):
                raise web.HTTPForbidden(text='Cross-origin requests are not allowed.')
            if request.content_length is not None and request.content_length > 8192:
                raise web.HTTPRequestEntityTooLarge(max_size=8192, actual_size=request.content_length)
            if request.content_type != 'application/x-www-form-urlencoded':
                raise web.HTTPUnsupportedMediaType()
            body = bytearray()
            while True:
                chunk = await request.content.read(8193 - len(body))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > 8192:
                    raise web.HTTPRequestEntityTooLarge(max_size=8192, actual_size=len(body))
            try:
                fields = parse_qs(body.decode('utf-8'), keep_blank_values=True,
                                  strict_parsing=True, max_num_fields=2)
                valid = (set(fields) == {'token'} and len(fields['token']) == 1
                         and hmac.compare_digest(fields['token'][0].encode('utf-8'), auth.secret.encode('utf-8')))
            except (ValueError, UnicodeError):
                valid = False
            if not valid:
                return unauthorized(request)
            return with_session(web.Response(status=303, headers={
                'Location': '/', 'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'}))
        token_present = 'token' in request.query
        token_valid = (native_token and request.method == 'GET'
                       and len(request.query.getall('token', [])) == 1
                       and hmac.compare_digest(request.query['token'].encode('utf-8'), auth.secret.encode('utf-8')))
        if token_present and not token_valid:
            return unauthorized(request)
        authenticated = (token_valid or session_authorized(request.cookies.get(SESSION_COOKIE), auth.secret)
                         if native_token else authorized(request.headers.get('Authorization', ''), auth.secret))
        if not authenticated:
            return unauthorized(request)
        origin = request.headers.get('Origin')
        if not origin_allowed(origin, request.host, public_origin):
            raise web.HTTPForbidden(text='Cross-origin requests are not allowed.')
        if native_token and request.path == '/api/status':
            if request.method != 'GET':
                raise web.HTTPMethodNotAllowed(request.method, ['GET'])
            return web.json_response({'ready': bool(state.get('ready')),
                                      'phase': state.get('phase', 'starting')},
                                     headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
        if token_valid or (native_token and request.path in ('/lab', '/lab/')):
            destination = request.rel_url.with_query([(key, value) for key, value in request.query.items() if key != 'token'])
            if request.path in ('/lab', '/lab/') or request.path.startswith('//') or '\\' in request.path:
                destination = destination.with_path('/').with_query(destination.query)
            if token_valid:
                # A cross-site Connect navigation does not carry a newly set
                # Strict cookie across an HTTP redirect. Complete a same-origin
                # document first, replace the token URL in history, then open
                # the clean URL with the browser's protected session cookie.
                nonce = secrets.token_urlsafe(18)
                encoded_destination = (json.dumps(str(destination)).replace('<', '\\u003c')
                                       .replace('>', '\\u003e').replace('&', '\\u0026'))
                transition = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Opening Fibo Edit 2 Studio</title>
<p>Opening Fibo Edit 2 Studio…</p><a href="{html.escape(str(destination), quote=True)}">Continue to Fibo Edit 2 Studio</a>
<script nonce="{nonce}">const destination={encoded_destination};history.replaceState(null,'',destination);location.replace(destination);</script></html>'''
                return with_session(web.Response(text=transition, content_type='text/html', headers={
                    'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                    'Content-Security-Policy': "default-src 'none'; script-src 'nonce-" + nonce
                    + "'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"}))
            response = web.Response(status=303, headers={'Location': str(destination),
                'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer'})
            return response
        if request.path == '/fibo/status':
            return web.json_response({k: state.get(k) for k in ('phase', 'detail', 'ready', 'bytes_total', 'bytes_complete')},
                                     headers={'Cache-Control': 'no-store'})
        if not state.get('ready') or (request.path == '/' and not any(k in request.query for k in ('template', 'editor'))):
            if request.method != 'GET':
                return web.json_response({'error': 'Model preparation is still in progress.'}, status=503)
            return web.Response(text=page(state, config), content_type='text/html', headers={'Cache-Control': 'no-store'})
        target = upstream.rstrip('/') + str(request.rel_url.with_query(
            [(key, value) for key, value in request.query.items() if key != 'token']))
        session = request.app[UPSTREAM_SESSION]
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            server = await session.ws_connect(target, max_msg_size=64 * 1024**2)
            browser = web.WebSocketResponse(max_msg_size=64 * 1024**2)
            await browser.prepare(request)

            async def pump(source, destination):
                async for message in source:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await destination.send_str(message.data)
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        await destination.send_bytes(message.data)
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        break
            tasks = [asyncio.create_task(pump(server, browser)), asyncio.create_task(pump(browser, server))]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await server.close()
                await browser.close()
            return browser
        # Browser origin checks belong at this public boundary.
        # ComfyUI's loopback origin guard would reject the public browser origin
        # after the proxy replaces Host with its private upstream address.
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in HOP_HEADERS and k.lower() != 'origin'
                   and k.lower() not in ('cookie', 'referer')
                   and not k.lower().startswith('sec-fetch-')}
        async with session.request(request.method, target, headers=headers, data=request.content,
                                   allow_redirects=False) as response:
            outgoing = web.StreamResponse(status=response.status,
                headers={k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS})
            outgoing.headers['Referrer-Policy'] = 'no-referrer'
            await outgoing.prepare(request)
            async for chunk in response.content.iter_chunked(64 * 1024):
                await outgoing.write(chunk)
            await outgoing.write_eof()
            return outgoing

    app.router.add_route('*', '/{path:.*}', dispatch)
    return app
