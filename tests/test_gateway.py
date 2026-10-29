import asyncio
import base64
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from fibo_runtime.gateway import (create_app, authorized, normalize_origin,
                                 public_origin_from_environment, origin_allowed,
                                 auth_from_environment, GatewayAuth, SESSION_COOKIE, SESSION_TTL,
                                 session_cookie, session_authorized)

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = 'test-password-for-gateway'
AUTH = {'Authorization': 'Basic ' + base64.b64encode(('fibo:' + PASSWORD).encode()).decode()}
POD_ID = 'releasepod1234'
POD_ORIGIN = 'https://' + POD_ID + '-8188.proxy.runpod.net'


def test_origin_normalization_and_invalid_values():
    assert normalize_origin('HTTPS://EXAMPLE.COM:443') == ('https', 'example.com', 443)
    assert normalize_origin('http://localhost:80') == ('http', 'localhost', 80)
    assert normalize_origin('http://[::1]:8188') == ('http', '::1', 8188)
    assert normalize_origin('http://[0:0:0:0:0:0:0:1]:8188') == ('http', '::1', 8188)
    assert normalize_origin('https://bücher.example') == ('https', 'xn--bcher-kva.example', 443)
    assert origin_allowed('https://example.com', 'EXAMPLE.COM:443', None)
    for invalid in ('null', '', 'https://user:secret@example.com', 'https://example.com/',
                    'https://example.com/path', 'https://example.com?x=1',
                    'https://example.com#fragment', 'https://example.com?',
                    'https://example.com#', ' https://example.com',
                    'https://example.com\n', 'https://one.example https://two.example',
                    'https://example.com:0', 'https://example.com:65536',
                    'https://example.com:', 'https://example.com\\evil', 'file://example.com',
                    'https://[broken', 'https://example.com,evil.example'):
        assert normalize_origin(invalid) is None, invalid
        assert not origin_allowed(invalid, 'example.com', None)


def test_public_origin_configuration_is_explicit_and_validated(monkeypatch):
    monkeypatch.delenv('FIBO_PUBLIC_ORIGIN', raising=False)
    monkeypatch.setenv('RUNPOD_POD_ID', POD_ID)
    expected = ('https', 'releasepod1234-8188.proxy.runpod.net', 443)
    assert public_origin_from_environment() == expected
    assert origin_allowed(POD_ORIGIN + ':443', '10.2.3.4:8188', expected)
    assert not origin_allowed(POD_ORIGIN.replace('https:', 'http:'), '10.2.3.4:8188', expected)
    assert not origin_allowed('http://10.2.3.4:8188', '10.2.3.4:8188', expected)
    monkeypatch.setenv('FIBO_PUBLIC_ORIGIN', 'https://studio.example:8443')
    assert public_origin_from_environment() == ('https', 'studio.example', 8443)
    monkeypatch.setenv('FIBO_PUBLIC_ORIGIN', 'https://studio.example/unexpected-path')
    with pytest.raises(ValueError, match='FIBO_PUBLIC_ORIGIN'):
        create_app({'ready': True}, ROOT, PASSWORD)
    monkeypatch.delenv('FIBO_PUBLIC_ORIGIN')
    monkeypatch.setenv('RUNPOD_POD_ID', 'bad/id')
    with pytest.raises(ValueError, match='RUNPOD_POD_ID'):
        create_app({'ready': True}, ROOT, PASSWORD)
    monkeypatch.setenv('RUNPOD_POD_ID', 'a' * 59)
    with pytest.raises(ValueError, match='RUNPOD_POD_ID'):
        create_app({'ready': True}, ROOT, PASSWORD)


def test_runpod_rewritten_host_allows_own_browser_post_and_websocket(monkeypatch):
    monkeypatch.delenv('FIBO_PUBLIC_ORIGIN', raising=False)
    monkeypatch.delenv('FIBO_PASSWORD', raising=False)
    monkeypatch.setenv('JUPYTER_PASSWORD', 'separate-jupyter-token')
    monkeypatch.setenv('RUNPOD_POD_ID', POD_ID)

    async def run():
        seen = []
        async def endpoint(request):
            seen.append(dict(request.headers))
            if request.path == '/ws':
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                async for message in ws:
                    await ws.send_str(message.data)
                return ws
            return web.json_response({'accepted': await request.json()})
        upstream = web.Application()
        upstream.router.add_route('*', '/{path:.*}', endpoint)
        async with TestServer(upstream) as server:
            app = create_app({'ready': True}, ROOT, PASSWORD, upstream=str(server.make_url('')).rstrip('/'))
            async with TestClient(TestServer(app)) as client:
                headers = {**AUTH, 'Host': '10.2.3.4:8188',
                           'Origin': POD_ORIGIN}
                response = await client.post('/prompt', headers=headers, json={'owned_test': True})
                assert response.status == 200
                assert (await response.json())['accepted'] == {'owned_test': True}
                async with client.ws_connect('/ws?clientId=owned-test', headers=headers) as ws:
                    await ws.send_str('owned event')
                    assert (await asyncio.wait_for(ws.receive(), 2)).data == 'owned event'
                for origin in ('https://attacker.example', 'http://10.2.3.4:8188', 'null'):
                    spoofed = {**headers, 'Origin': origin, 'X-Forwarded-Host': 'attacker.example',
                               'Forwarded': 'proto=https;host=attacker.example'}
                    assert (await client.post('/prompt', headers=spoofed, json={})).status == 403
                    with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                        await client.ws_connect('/ws?clientId=blocked-test', headers=spoofed)
                    assert error.value.status == 403
        assert len(seen) == 2
        assert all('Authorization' not in headers and 'Origin' not in headers for headers in seen)
    asyncio.run(run())


def test_unicode_password_and_invalid_login():
    secret = 'valid-unicode-password-🔐'
    header = 'Basic ' + base64.b64encode(('fibo:' + secret).encode()).decode()
    assert authorized(header, secret)
    assert not authorized(header, PASSWORD)
    bad_user = 'Basic ' + base64.b64encode('móss:incorrect'.encode()).decode()
    assert not authorized(bad_user, secret)


def test_gateway_auth_configuration_and_bounded_signed_sessions(monkeypatch):
    monkeypatch.delenv('FIBO_PASSWORD', raising=False)
    monkeypatch.delenv('JUPYTER_PASSWORD', raising=False)
    with pytest.raises(ValueError, match='ComfyUI authentication is missing'):
        auth_from_environment()
    monkeypatch.setenv('JUPYTER_PASSWORD', PASSWORD)
    auth = auth_from_environment()
    assert auth == GatewayAuth('token', PASSWORD, 8188)
    monkeypatch.setenv('FIBO_PASSWORD', 'legacy-unique-password')
    assert auth_from_environment().mechanism == 'basic'
    assert auth_from_environment().port == 8188
    for invalid in ('short', '{{ RUNPOD_SECRET_missing_password }}', 'password-with-newline\n'):
        monkeypatch.setenv('FIBO_PASSWORD', invalid)
        with pytest.raises(ValueError):
            auth_from_environment()
    monkeypatch.setenv('FIBO_PASSWORD', '')
    monkeypatch.setenv('JUPYTER_PASSWORD', '{{ RUNPOD_SECRET_missing_token }}')
    with pytest.raises(ValueError, match='unresolved Runpod Secret'):
        auth_from_environment()
    assert auth_from_environment('serverless') == GatewayAuth('none', '', 8188)

    value = session_cookie(PASSWORD, now=1000)
    assert session_authorized(value, PASSWORD, now=1000)
    assert session_authorized(value, PASSWORD, now=1000 + SESSION_TTL - 1)
    assert not session_authorized(value, PASSWORD, now=999)
    assert not session_authorized(value, PASSWORD, now=1000 + SESSION_TTL)
    assert not session_authorized(value, 'a-different-valid-password', now=1000)
    assert not session_authorized(value[:-1] + ('0' if value[-1] != '0' else '1'), PASSWORD, now=1000)
    assert not session_authorized('invalid', PASSWORD, now=1000)


def test_entrypoint_passes_only_jupyter_token_to_base_service(tmp_path):
    import os
    import shlex
    import subprocess
    import sys

    # Substitute only external executable paths. Run the actual shell control
    # flow and auth preflight, with a fake SSH/base hook and GPU-free gateway.
    base = tmp_path / 'base-start'
    base.write_text(f'#!{sys.executable}\n' + '''import json, os
from pathlib import Path
keys = ('FIBO_PASSWORD', 'JUPYTER_PASSWORD', 'JUPYTER_DISABLE_AUTH')
Path(os.environ['FIBO_TEST_BASE_RECORD']).write_text(json.dumps({key: key in os.environ for key in keys}))
Path(os.environ['FIBO_START_SIGNAL']).write_text('ready')
''')
    python = tmp_path / 'gateway-python'
    python.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
if sys.argv[1:] == ['-']:
    os.execv(sys.executable, [sys.executable, '-'])
assert sys.argv[1:] == ['-m', 'fibo_runtime.bootstrap']
keys = ('FIBO_PASSWORD', 'JUPYTER_PASSWORD', 'JUPYTER_DISABLE_AUTH')
Path(os.environ['FIBO_TEST_PARENT_RECORD']).write_text(json.dumps({key: key in os.environ for key in keys}))
''')
    for executable in (base, python):
        executable.chmod(0o755)
    entrypoint = tmp_path / 'entrypoint.sh'
    entrypoint.write_text((ROOT / 'scripts/entrypoint.sh').read_text()
                         .replace('/opt/venv/bin/python', shlex.quote(str(python)))
                         .replace('/start.sh', shlex.quote(str(base))))
    environment = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'FIBO_MODE': 'pod',
                   'FIBO_PASSWORD': PASSWORD, 'JUPYTER_PASSWORD': 'provider-test-connect-token',
                   'JUPYTER_DISABLE_AUTH': 'true',
                   'FIBO_TEST_BASE_RECORD': str(tmp_path / 'base.json'),
                   'FIBO_TEST_PARENT_RECORD': str(tmp_path / 'parent.json')}
    result = subprocess.run(['bash', str(entrypoint)], env=environment, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / 'base.json').read_text()) == {
        'FIBO_PASSWORD': False, 'JUPYTER_PASSWORD': True, 'JUPYTER_DISABLE_AUTH': False}
    assert json.loads((tmp_path / 'parent.json').read_text()) == {
        'FIBO_PASSWORD': True, 'JUPYTER_PASSWORD': True, 'JUPYTER_DISABLE_AUTH': True}


def test_gateway_form_login_keeps_token_out_of_urls_and_checks_origin(monkeypatch):
    monkeypatch.delenv('FIBO_PASSWORD', raising=False)
    monkeypatch.delenv('FIBO_PUBLIC_ORIGIN', raising=False)
    monkeypatch.setenv('JUPYTER_PASSWORD', PASSWORD)
    monkeypatch.setenv('RUNPOD_POD_ID', POD_ID)

    async def run():
        state = {'ready': False, 'phase': 'downloading'}
        async with TestClient(TestServer(create_app(state, ROOT))) as client:
            response = await client.get('/')
            form = await response.text()
            assert response.status == 401
            assert '<form method="post" action="/fibo/login">' in form
            assert 'name="token" type="password"' in form
            assert PASSWORD not in form and '?token=' not in form
            assert response.headers['Cache-Control'] == 'no-store'
            assert response.headers['Referrer-Policy'] == 'no-referrer'
            assert "form-action 'self'" in response.headers['Content-Security-Policy']
            assert 'Set-Cookie' not in response.headers
            headers = {'Origin': POD_ORIGIN}
            response = await client.post('/fibo/login', data={'token': PASSWORD},
                                         headers={'Origin': 'https://attacker.example'}, allow_redirects=False)
            assert response.status == 403 and 'Set-Cookie' not in response.headers
            for data in ({'token': 'wrong-token'}, [('token', PASSWORD), ('token', PASSWORD)]):
                response = await client.post('/fibo/login', data=data, headers=headers, allow_redirects=False)
                assert response.status == 401 and 'Set-Cookie' not in response.headers
                assert PASSWORD not in await response.text()
            response = await client.post('/fibo/login', data=b'token=' + b'x' * 8192,
                                         headers={**headers, 'Content-Type': 'application/x-www-form-urlencoded'},
                                         allow_redirects=False)
            assert response.status == 413 and 'Set-Cookie' not in response.headers

            async def oversized_chunks():
                yield b'token=' + b'x' * 4096
                yield b'x' * 4096
            response = await client.post('/fibo/login', data=oversized_chunks(),
                                         headers={**headers, 'Content-Type': 'application/x-www-form-urlencoded'},
                                         allow_redirects=False)
            assert response.status == 413 and 'Set-Cookie' not in response.headers
            response = await client.post('/fibo/login', data={'token': PASSWORD},
                                         headers=headers, allow_redirects=False)
            assert response.status == 303 and response.headers['Location'] == '/'
            assert PASSWORD not in await response.text()
            assert response.headers['Cache-Control'] == 'no-store'
            assert response.headers['Referrer-Policy'] == 'no-referrer'
            cookie = response.cookies[SESSION_COOKIE]
            assert cookie['secure'] and cookie['httponly'] and cookie['samesite'] == 'Strict'
            assert cookie['path'] == '/' and not cookie['domain']
            assert PASSWORD not in cookie.value
            # TestServer is HTTP; supply the HTTPS-only cookie explicitly.
            response = await client.get('/fibo/status', headers={
                'Cookie': SESSION_COOKIE + '=' + cookie.value, 'Origin': POD_ORIGIN})
            assert response.status == 200 and (await response.json())['ready'] is False
    asyncio.run(run())


def test_explicit_token_exchange_preparation_and_cookie_transport(monkeypatch):
    monkeypatch.delenv('FIBO_PASSWORD', raising=False)
    monkeypatch.delenv('FIBO_PUBLIC_ORIGIN', raising=False)
    monkeypatch.setenv('JUPYTER_PASSWORD', PASSWORD)
    monkeypatch.setenv('RUNPOD_POD_ID', POD_ID)
    native_origin = POD_ORIGIN
    assert public_origin_from_environment(8188) == ('https', POD_ID + '-8188.proxy.runpod.net', 443)

    async def run():
        state = {'ready': False, 'phase': 'downloading', 'detail': 'Private workspace details'}
        seen = []
        async def endpoint(request):
            seen.append((str(request.rel_url), dict(request.headers)))
            if request.path == '/ws':
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                async for message in ws:
                    await ws.send_str(message.data)
                return ws
            return web.json_response({'accepted': await request.json()})
        upstream = web.Application()
        upstream.router.add_route('*', '/{path:.*}', endpoint)
        async with TestServer(upstream) as server:
            app = create_app(state, ROOT, upstream=str(server.make_url('')).rstrip('/'),
                             auth=GatewayAuth('token', PASSWORD, 8188))
            async with TestClient(TestServer(app)) as client:
                for path in ('/', '/lab', '/api/status', '/ws', '/object_info'):
                    assert (await client.get(path, allow_redirects=False)).status == 401
                for suffix in ('?token=invalid', '?token=' + PASSWORD + '&token=' + PASSWORD):
                    assert (await client.get('/lab' + suffix, allow_redirects=False)).status == 401
                # Runpod's Connect probe is healthy throughout model preparation,
                # but only after its platform-issued token has been authenticated.
                response = await client.get('/api/status', params={'token': PASSWORD})
                assert response.status == 200
                assert await response.json() == {'ready': False, 'phase': 'downloading'}
                assert 'Private workspace details' not in await response.text()
                assert 'Set-Cookie' not in response.headers
                assert (await client.get('/api/status', params={'token': PASSWORD},
                    headers={'Origin': 'https://console.runpod.io'})).status == 403
                for path in ('/', '/lab', '/lab/'):
                    response = await client.get(path, params={'token': PASSWORD, 'editor': '1'}, allow_redirects=False)
                    assert response.status == 200
                    assert 'Location' not in response.headers
                    transition = await response.text()
                    assert 'const destination="/?editor=1";' in transition
                    assert "history.replaceState(null,'',destination);location.replace(destination)" in transition
                    assert 'href="/?editor=1"' in transition
                    assert PASSWORD not in transition
                    assert 'token=' not in transition
                    nonce = re.search(r'<script nonce="([A-Za-z0-9_-]+)">', transition).group(1)
                    assert "script-src 'nonce-" + nonce + "'" in response.headers['Content-Security-Policy']
                    assert "default-src 'none'" in response.headers['Content-Security-Policy']
                    assert response.headers['Cache-Control'] == 'no-store'
                    assert response.headers['Referrer-Policy'] == 'no-referrer'
                    cookie = response.cookies[SESSION_COOKIE]
                    assert cookie['secure'] and cookie['httponly'] and cookie['samesite'] == 'Strict'
                    assert cookie['path'] == '/' and int(cookie['max-age']) == SESSION_TTL
                    assert SESSION_COOKIE.startswith('__Host-') and not cookie['domain']
                attack = '</script><script>alert("unsafe")</script>&other="value"'
                response = await client.get('/lab', params={'token': PASSWORD, 'editor': attack, 'owned': '1'}, allow_redirects=False)
                transition = await response.text()
                assert response.status == 200 and transition.count('<script ') == 1
                assert transition.count('</script>') == 1 and attack not in transition
                assert PASSWORD not in transition and 'token=' not in transition
                encoded_url = re.search(r'const destination=(.*?);history\.', transition).group(1)
                clean_url = json.loads(encoded_url)
                assert clean_url.startswith('/?editor=') and not clean_url.startswith('//')
                assert '\\u0026' in encoded_url
                # TestServer is HTTP, so supply the Secure browser cookie
                # explicitly; production browsers send it over the HTTPS proxy.
                headers = {'Cookie': SESSION_COOKIE + '=' + cookie.value,
                           'Host': '10.2.3.4:8188', 'Origin': native_origin}
                response = await client.get('/lab?editor=1', headers=headers, allow_redirects=False)
                assert response.status == 303 and response.headers['Location'] == '/?editor=1'
                assert (await client.get('/fibo/status', headers=headers)).status == 200
                assert (await client.get('/api/status?token=invalid', headers=headers)).status == 401
                assert (await client.post('/prompt', headers=headers, json={})).status == 503
                state.update(ready=True, phase='ready')
                response = await client.post('/prompt?owned=1', headers={**headers,
                    'Referer': native_origin + '/?token=' + PASSWORD}, json={'owned_test': True})
                assert response.status == 200
                assert (await response.json())['accepted'] == {'owned_test': True}
                async with client.ws_connect('/ws?clientId=owned-native-test', headers=headers) as ws:
                    await ws.send_str('owned native event')
                    assert (await asyncio.wait_for(ws.receive(), 2)).data == 'owned native event'
                foreign = {**headers, 'Origin': 'https://attacker.example',
                           'X-Forwarded-Host': 'attacker.example'}
                assert (await client.post('/prompt', headers=foreign, json={})).status == 403
                with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                    await client.ws_connect('/ws', headers=foreign)
                assert error.value.status == 403
                tampered = cookie.value[:-1] + ('0' if cookie.value[-1] != '0' else '1')
                assert (await client.get('/fibo/status', headers={'Cookie': SESSION_COOKIE + '=' + tampered})).status == 401
        assert len(seen) == 2
        assert seen[0][0] == '/prompt?owned=1'
        assert all('token=' not in url for url, _ in seen)
        assert all(not ({'Authorization', 'Cookie', 'Referer', 'Origin'} & set(headers)) for _, headers in seen)
    asyncio.run(run())


def test_preparation_auth_and_serverless_isolation():
    async def run():
        state = {'ready': False, 'phase': 'downloading', 'detail': 'Preparing models'}
        async with TestClient(TestServer(create_app(state, ROOT, PASSWORD))) as client:
            assert (await client.get('/healthz')).status == 200
            assert (await client.get('/readyz')).status == 503
            assert (await client.get('/ping')).status == 204
            for path in ('/', '/fibo/status', '/object_info', '/ws', '/view?filename=private.png'):
                assert (await client.get(path)).status == 401
            assert (await client.get('/', headers={'Authorization': 'Basic !!bad'})).status == 401
            response = await client.get('/', headers=AUTH)
            assert 'Downloading models' in await response.text()
            assert (await client.post('/prompt', headers=AUTH, json={})).status == 503
            state.update(ready=True, phase='ready')
            assert (await client.get('/readyz')).status == 200
            html = await (await client.get('/', headers=AUTH)).text()
            assert '?template=fibo2-edit-turbo&amp;source=fibo' in html
            assert html.count('Open workflow') == 1
            assert 'http-equiv="refresh"' not in html
            assert (await client.get('/', headers={**AUTH, 'Origin': 'https://unrelated.example'})).status == 403
        async with TestClient(TestServer(create_app(state, ROOT, '', mode='serverless'))) as client:
            assert (await client.get('/readyz')).status == 200
            assert (await client.get('/', headers=AUTH)).status == 404
            assert (await client.get('/object_info', headers=AUTH)).status == 404
    asyncio.run(run())


def test_proxy_http_websocket_and_credentials_stay_at_gateway():
    async def run():
        # Reproduce the relevant pinned ComfyUI origin middleware behavior:
        # browser Origin must match a loopback Host, including explicit ports.
        @web.middleware
        async def comfy_origin_guard(request, handler):
            if request.headers.get('Sec-Fetch-Site') == 'cross-site':
                return web.Response(status=403)
            if 'Host' in request.headers and 'Origin' in request.headers:
                host = urlsplit('//' + request.headers['Host'].lower())
                origin = urlsplit(request.headers['Origin'].lower())
                host_domain = host.netloc if origin.port is not None else host.hostname
                origin_domain = origin.netloc if host.port is not None else origin.hostname
                if host.hostname in ('127.0.0.1', 'localhost', '::1') and host_domain != origin_domain:
                    return web.Response(status=403)
            return await handler(request)

        upstream = web.Application(middlewares=[comfy_origin_guard])
        seen = []
        async def endpoint(request):
            seen.append(dict(request.headers))
            if request.path == '/ws':
                ws = web.WebSocketResponse()
                await ws.prepare(request)
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        await ws.send_str(message.data)
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        await ws.send_bytes(message.data)
                return ws
            return web.json_response({'path': str(request.rel_url), 'body': await request.text()})
        upstream.router.add_route('*', '/{path:.*}', endpoint)
        async with TestServer(upstream) as server:
            app = create_app({'ready': True}, ROOT, PASSWORD, upstream=str(server.make_url('')).rstrip('/'))
            async with TestClient(TestServer(app)) as client:
                browser_headers = {**AUTH, 'Host': 'example-8188.proxy.runpod.net',
                                   'Origin': 'https://example-8188.proxy.runpod.net',
                                   'Sec-Fetch-Site': 'same-origin', 'Sec-Fetch-Mode': 'cors'}
                async with client.post('/prompt?test=1', headers=browser_headers, json={'test': 42}) as response:
                    assert response.status == 200
                    result = await response.json()
                    assert result['path'] == '/prompt?test=1'
                    assert json.loads(result['body']) == {'test': 42}
                assert (await client.post('/prompt', headers={**browser_headers,
                    'Origin': 'https://unrelated.example'}, json={'test': 42})).status == 403
                async with client.ws_connect('/ws?clientId=test', headers=AUTH) as ws:
                    await ws.send_str('progress')
                    assert (await asyncio.wait_for(ws.receive(), 2)).data == 'progress'
                    await ws.send_bytes(b'preview')
                    assert (await asyncio.wait_for(ws.receive(), 2)).data == b'preview'
        assert len(seen) == 2
        assert all('Authorization' not in headers for headers in seen)
        assert all('Origin' not in headers for headers in seen)
        assert all(not name.lower().startswith('sec-fetch-') for headers in seen for name in headers)
    asyncio.run(run())
