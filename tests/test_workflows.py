import json
from pathlib import Path

from fibo_runtime.workflows import install_workflows


def bundle(root, text='first'):
    (root / 'config').mkdir(exist_ok=True)
    (root / 'workflows/ui').mkdir(parents=True, exist_ok=True)
    (root / 'workflows/ui/test.json').write_text(json.dumps({'text': text}))
    (root / 'config/workflows.json').write_text(json.dumps({'presets': {'test': {'ui': 'workflows/ui/test.json'}}}))


def test_upgrade_preserves_edits_and_updates_pristine_files(tmp_path):
    source, data = tmp_path/'package', tmp_path/'data'
    source.mkdir()
    bundle(source)
    target = data/'user/default/workflows/FIBO/test.json'
    assert install_workflows(source, data)['installed']
    bundle(source, 'updated')
    assert install_workflows(source, data)['updated']
    assert json.loads(target.read_text())['text'] == 'updated'
    target.write_text('{"text":"my changes"}')
    bundle(source, 'new release')
    assert install_workflows(source, data)['preserved']
    assert json.loads(target.read_text())['text'] == 'my changes'


def test_existing_link_is_not_followed(tmp_path):
    source, data = tmp_path/'package', tmp_path/'data'
    source.mkdir()
    bundle(source)
    outside = tmp_path/'outside.json'
    outside.write_text('user content')
    target = data/'user/default/workflows/FIBO/test.json'
    target.parent.mkdir(parents=True)
    target.symlink_to(outside)
    assert install_workflows(source, data)['preserved']
    assert outside.read_text() == 'user content'
