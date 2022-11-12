/**
 * @license
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import 'jasmine';

import {AttentionHeads, BooleanLitType, CategoryLabel, Embeddings, MulticlassPreds, Scalar, TextSegment, TokenGradients, Tokens} from './lit_types';
import {LitMetadata, SerializedLitMetadata} from './types';
import {createLitType} from './utils';

/**
 * Cleans state between tests.
 */
export function cleanState<State extends {}>(
    init: null|(() => Promise<State>)|(() => State) = null,
    beforeEachCustom:
        (action: (done: DoneFn) => void, timeout?: number|undefined) => void =
            beforeEach) {
  const state = {} as State;
  beforeEachCustom(async () => {
    // Clear state before every test case.
    for (const prop of Object.getOwnPropertyNames(state)) {
      // tslint:disable-next-line:no-any Dynamically accessing state properties.
      delete (state as {[k: string]: any})[prop];
    }
    if (init) {
      Object.assign(state, await init());
    }
  });

  return state;
}

function emptySpec() {
  return {'configSpec': {}, 'metaSpec': {}};
}

/**
 * Mock metadata describing a set of models, datasets, generators, and
 * intepretators.
 */
export const mockMetadata: LitMetadata = {
  'models': {
    'sst_0_micro': {
      'spec': {
        'input': {
          'passage': createLitType(TextSegment),
          'passage_tokens':
              createLitType(Tokens, {'required': false, 'parent': 'passage'}),
        },
        'output': {
          'probabilities': createLitType(
              MulticlassPreds,
              {'vocab': ['0', '1'], 'null_idx': 0, 'parent': 'label'}),
          'pooled_embs': createLitType(Embeddings),
          'mean_word_embs': createLitType(Embeddings),
          'tokens': createLitType(Tokens),
          'passage_tokens': createLitType(Tokens, {'parent': 'passage'}),
          'passage_grad':
              createLitType(TokenGradients, {'align': 'passage_tokens'}),
          'layer_0/attention': createLitType(AttentionHeads, {
            'align_in': 'tokens',
            'align_out': 'tokens',
          }),
          'layer_1/attention': createLitType(AttentionHeads, {
            'align_in': 'tokens',
            'align_out': 'tokens',
          }),
        }
      },
      'datasets': ['sst_dev'],
      'generators':
          ['word_replacer', 'scrambler', 'backtranslation', 'hotflip'],
      'interpreters':
          ['grad_norm', 'grad_sum', 'lime', 'metrics', 'pca', 'umap']
    },
    'sst_1_micro': {
      'spec': {
        'input': {
          'passage': createLitType(TextSegment),
          'passage_tokens':
              createLitType(Tokens, {'required': false, 'parent': 'passage'}),
        },
        'output': {
          'probabilities': createLitType(
              MulticlassPreds,
              {'vocab': ['0', '1'], 'null_idx': 0, 'parent': 'label'}),
          'pooled_embs': createLitType(Embeddings),
          'mean_word_embs': createLitType(Embeddings),
          'tokens': createLitType(Tokens),
          'passage_tokens': createLitType(Tokens, {'parent': 'passage'}),
          'passage_grad':
              createLitType(TokenGradients, {'align': 'passage_tokens'}),
          'layer_0/attention': createLitType(AttentionHeads, {
            'align_in': 'tokens',
            'align_out': 'tokens',
          }),
          'layer_1/attention': createLitType(AttentionHeads, {
            'align_in': 'tokens',
            'align_out': 'tokens',
          })
        }
      },
      'datasets': ['sst_dev'],
      'generators':
          ['word_replacer', 'scrambler', 'backtranslation', 'hotflip'],
      'interpreters':
          ['grad_norm', 'grad_sum', 'lime', 'metrics', 'pca', 'umap']
    }
  },
  'datasets': {
    'sst_dev': {
      'size': 872,
      'spec': {
        'passage': createLitType(TextSegment),
        'label': createLitType(CategoryLabel, {'vocab': ['0', '1']}),
      }
    },
    'color_test': {
      'size': 2,
      'spec': {
        'testNumFeat0': createLitType(Scalar),
        'testNumFeat1': createLitType(Scalar),
        'testFeat0': createLitType(CategoryLabel, {'vocab': ['0', '1']}),
        'testFeat1': createLitType(CategoryLabel, {'vocab': ['a', 'b', 'c']})
      }
    },
    'penguin_dev': {
      'size': 10,
      'spec': {
        'body_mass_g': createLitType(Scalar, {
          'step': 1,
        }),
        'culmen_depth_mm': createLitType(Scalar, {
          'step': 1,
        }),
        'culmen_length_mm': createLitType(Scalar, {
          'step': 1,
        }),
        'flipper_length_mm': createLitType(Scalar, {
          'step': 1,
        }),
        'island': createLitType(
            CategoryLabel, {'vocab': ['Biscoe', 'Dream', 'Torgersen']}),
        'sex': createLitType(CategoryLabel, {'vocab': ['female', 'male']}),
        'species': createLitType(
            CategoryLabel, {'vocab': ['Adelie', 'Chinstrap', 'Gentoo']}),
        'isAlive': createLitType(BooleanLitType, {'required': false})
      }
    }
  },
  'generators': {
    'word_replacer': {
      'configSpec': {
        'Substitutions':
            createLitType(TextSegment, {'default': 'great -> terrible'}),
      },
      'metaSpec': {}
    },
    'scrambler': emptySpec(),
    'backtranslation': emptySpec(),
    'hotflip': emptySpec(),
  },
  'interpreters': {
    'grad_norm': emptySpec(),
    'grad_sum': emptySpec(),
    'lime': emptySpec(),
    'metrics': emptySpec(),
    'pca': emptySpec(),
    'umap': emptySpec(),
  },
  'layouts': {},
  'demoMode': false,
  'defaultLayout': 'default',
  'canonicalURL': undefined,
  'syncState': false
};

/**
 * Mock serialized metadata describing a set of models, datasets, generators,
 * and intepretators. Corresponds to the mockMetadata above.
 */
export const mockSerializedMetadata: SerializedLitMetadata = {
  'models': {
    'sst_0_micro': {
      'spec': {
        'input': {
          'passage': {'__name__': 'TextSegment', 'required': true},
          'passage_tokens':
              {'__name__': 'Tokens', 'required': false, 'parent': 'passage'}
        },
        'output': {
          'probabilities': {
            '__name__': 'MulticlassPreds',
            'required': true,
            'vocab': ['0', '1'],
            'null_idx': 0,
            'parent': 'label'
          },
          'pooled_embs': {'__name__': 'Embeddings', 'required': true},
          'mean_word_embs': {'__name__': 'Embeddings', 'required': true},
          'tokens':
              {'__name__': 'Tokens', 'required': true, 'parent': undefined},
          'passage_tokens':
              {'__name__': 'Tokens', 'required': true, 'parent': 'passage'},
          'passage_grad': {
            '__name__': 'TokenGradients',
            'required': true,
            'align': 'passage_tokens'
          },
          'layer_0/attention': {
            '__name__': 'AttentionHeads',
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          },
          'layer_1/attention': {
            '__name__': 'AttentionHeads',
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          }
        }
      },
      'datasets': ['sst_dev'],
      'generators':
          ['word_replacer', 'scrambler', 'backtranslation', 'hotflip'],
      'interpreters':
          ['grad_norm', 'grad_sum', 'lime', 'metrics', 'pca', 'umap']
    },
    'sst_1_micro': {
      'spec': {
        'input': {
          'passage': {'__name__': 'TextSegment', 'required': true},
          'passage_tokens':
              {'__name__': 'Tokens', 'required': false, 'parent': 'passage'}
        },
        'output': {
          'probabilities': {
            '__name__': 'MulticlassPreds',
            'required': true,
            'vocab': ['0', '1'],
            'null_idx': 0,
            'parent': 'label'
          },
          'pooled_embs': {'__name__': 'Embeddings', 'required': true},
          'mean_word_embs': {'__name__': 'Embeddings', 'required': true},
          'tokens':
              {'__name__': 'Tokens', 'required': true, 'parent': undefined},
          'passage_tokens':
              {'__name__': 'Tokens', 'required': true, 'parent': 'passage'},
          'passage_grad': {
            '__name__': 'TokenGradients',
            'required': true,
            'align': 'passage_tokens'
          },
          'layer_0/attention': {
            '__name__': 'AttentionHeads',
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          },
          'layer_1/attention': {
            '__name__': 'AttentionHeads',
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          }
        }
      },
      'datasets': ['sst_dev'],
      'generators':
          ['word_replacer', 'scrambler', 'backtranslation', 'hotflip'],
      'interpreters':
          ['grad_norm', 'grad_sum', 'lime', 'metrics', 'pca', 'umap']
    }
  },
  'datasets': {
    'sst_dev': {
      'size': 872,
      'spec': {
        'passage': {'__name__': 'TextSegment', 'required': true},
        'label':
            {'__name__': 'CategoryLabel', 'required': true, 'vocab': ['0', '1']}
      }
    },
    'color_test': {
      'size': 2,
      'spec': {
        'testNumFeat0': {'__name__': 'Scalar', 'required': true},
        'testNumFeat1': {'__name__': 'Scalar', 'required': true},
        'testFeat0': {
          '__name__': 'CategoryLabel',
          'required': true,
          'vocab': ['0', '1']
        },
        'testFeat1': {
          '__name__': 'CategoryLabel',
          'required': true,
          'vocab': ['a', 'b', 'c']
        }
      }
    },
    'penguin_dev': {
      'size': 10,
      'spec': {
        'body_mass_g': {'__name__': 'Scalar', 'step': 1, 'required': true},
        'culmen_depth_mm': {'__name__': 'Scalar', 'step': 1, 'required': true},
        'culmen_length_mm': {'__name__': 'Scalar', 'step': 1, 'required': true},
        'flipper_length_mm':
            {'__name__': 'Scalar', 'step': 1, 'required': true},
        'island': {
          '__name__': 'CategoryLabel',
          'required': true,
          'vocab': ['Biscoe', 'Dream', 'Torgersen']
        },
        'sex': {
          '__name__': 'CategoryLabel',
          'required': true,
          'vocab': ['female', 'male']
        },
        'species': {
          '__name__': 'CategoryLabel',
          'required': true,
          'vocab': ['Adelie', 'Chinstrap', 'Gentoo']
        },
        'isAlive': {'__name__': 'BooleanLitType', 'required': false}
      }
    }
  },
  'generators': {
    'word_replacer': {
      'configSpec': {
        'Substitutions': {
          '__name__': 'TextSegment',
          'required': true,
          'default': 'great -> terrible'
        }
      },
      'metaSpec': {}
    },
    'scrambler': emptySpec(),
    'backtranslation': emptySpec(),
    'hotflip': emptySpec(),
  },
  'interpreters': {
    'grad_norm': emptySpec(),
    'grad_sum': emptySpec(),
    'lime': emptySpec(),
    'metrics': emptySpec(),
    'pca': emptySpec(),
    'umap': emptySpec(),
  },
  'layouts': {},
  'demoMode': false,
  'defaultLayout': 'default',
  'canonicalURL': undefined,
  'syncState': false
};
