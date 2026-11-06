import test from 'node:test';
import assert from 'node:assert/strict';
import {getModelHeading} from '../../../strata_server/web/model-heading.js';

test('the model heading includes the engine, configuration and first model file', () => {
  assert.deepEqual(getModelHeading({name: 'Qwen3.8 Flash Next', engine: 'strata-q38.exe',
    config_path: '/configs/model.json', model_path: '/models/first-shard.gguf'}, 'API model ID'), {
    name: 'Qwen3.8 Flash Next',
    title: 'Engine: strata-q38.exe\nConfiguration: /configs/model.json\nModel: /models/first-shard.gguf'});
  assert.deepEqual(getModelHeading(null, 'Fallback'), {name: 'Fallback', title: ''});
});
