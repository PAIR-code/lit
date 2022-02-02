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
import {LitMetadata} from './types';

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
          'passage': {
            '__class__': 'LitType',
            '__name__': 'TextSegment',
            '__mro__': ['TextSegment', 'LitType', 'object'],
            'required': true
          },
          'passage_tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': false,
            'parent': 'passage'
          }
        },
        'output': {
          'probabilities': {
            '__class__': 'LitType',
            '__name__': 'MulticlassPreds',
            '__mro__': ['MulticlassPreds', 'LitType', 'object'],
            'required': true,
            'vocab': ['0', '1'],
            'null_idx': 0,
            'parent': 'label'
          },
          'pooled_embs': {
            '__class__': 'LitType',
            '__name__': 'Embeddings',
            '__mro__': ['Embeddings', 'LitType', 'object'],
            'required': true
          },
          'mean_word_embs': {
            '__class__': 'LitType',
            '__name__': 'Embeddings',
            '__mro__': ['Embeddings', 'LitType', 'object'],
            'required': true
          },
          'tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': true,
            'parent': undefined
          },
          'passage_tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': true,
            'parent': 'passage'
          },
          'passage_grad': {
            '__class__': 'LitType',
            '__name__': 'TokenGradients',
            '__mro__': ['TokenGradients', 'LitType', 'object'],
            'required': true,
            'align': 'passage_tokens'
          },
          'layer_0/attention': {
            '__class__': 'LitType',
            '__name__': 'AttentionHeads',
            '__mro__': ['AttentionHeads', 'LitType', 'object'],
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          },
          'layer_1/attention': {
            '__class__': 'LitType',
            '__name__': 'AttentionHeads',
            '__mro__': ['AttentionHeads', 'LitType', 'object'],
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
          'passage': {
            '__class__': 'LitType',
            '__name__': 'TextSegment',
            '__mro__': ['TextSegment', 'LitType', 'object'],
            'required': true
          },
          'passage_tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': false,
            'parent': 'passage'
          }
        },
        'output': {
          'probabilities': {
            '__class__': 'LitType',
            '__name__': 'MulticlassPreds',
            '__mro__': ['MulticlassPreds', 'LitType', 'object'],
            'required': true,
            'vocab': ['0', '1'],
            'null_idx': 0,
            'parent': 'label'
          },
          'pooled_embs': {
            '__class__': 'LitType',
            '__name__': 'Embeddings',
            '__mro__': ['Embeddings', 'LitType', 'object'],
            'required': true
          },
          'mean_word_embs': {
            '__class__': 'LitType',
            '__name__': 'Embeddings',
            '__mro__': ['Embeddings', 'LitType', 'object'],
            'required': true
          },
          'tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': true,
            'parent': undefined
          },
          'passage_tokens': {
            '__class__': 'LitType',
            '__name__': 'Tokens',
            '__mro__': ['Tokens', 'LitType', 'object'],
            'required': true,
            'parent': 'passage'
          },
          'passage_grad': {
            '__class__': 'LitType',
            '__name__': 'TokenGradients',
            '__mro__': ['TokenGradients', 'LitType', 'object'],
            'required': true,
            'align': 'passage_tokens'
          },
          'layer_0/attention': {
            '__class__': 'LitType',
            '__name__': 'AttentionHeads',
            '__mro__': ['AttentionHeads', 'LitType', 'object'],
            'required': true,
            'align_in': 'tokens',
            'align_out': 'tokens',
          },
          'layer_1/attention': {
            '__class__': 'LitType',
            '__name__': 'AttentionHeads',
            '__mro__': ['AttentionHeads', 'LitType', 'object'],
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
      'spec': {
        'passage': {
          '__class__': 'LitType',
          '__name__': 'TextSegment',
          '__mro__': ['TextSegment', 'LitType', 'object'],
          'required': true
        },
        'label': {
          '__class__': 'LitType',
          '__name__': 'CategoryLabel',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['0', '1']
        }
      }
    },
    'color_test': {
      'spec': {
        'testNumFeat0': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'required': true
        },
        'testNumFeat1': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'required': true
        },
        'testFeat0': {
          '__class__': 'LitType',
          '__name__': 'CategoryLabel',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['0', '1']
        },
        'testFeat1': {
          '__class__': 'LitType',
          '__name__': 'CategoryLabel',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['a', 'b', 'c']
        }
      }
    },
    'penguin_dev': {
      'spec': {
        'body_mass_g': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'step': 1,
          'required': true
        },
        'culmen_depth_mm': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'step': 1,
          'required': true
        },
        'culmen_length_mm': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'step': 1,
          'required': true
        },
        'flipper_length_mm': {
          '__class__': 'LitType',
          '__name__': 'Scalar',
          '__mro__': ['Scalar', 'LitType', 'object'],
          'step': 1,
          'required': true
        },
        'island': {
          '__class__': 'LitType',
          '__name__': 'CategoryLabel',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['Biscoe', 'Dream', 'Torgersen']
        },
        'sex': {
          '__class__': 'LitType',
          '__name__': 'Boolean',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['female', 'male']
        },
        'species': {
          '__class__': 'LitType',
          '__name__': 'CategoryLabel',
          '__mro__': ['CategoryLabel', 'LitType', 'object'],
          'required': true,
          'vocab': ['Adelie', 'Chinstrap', 'Gentoo']
        },
        'isAlive': {
          '__class__': 'LitType',
          '__name__': 'Boolean',
          '__mro__': ['Boolean', 'LitType', 'object'],
          'required': false
        }
      }
    }
  },
  'generators': {
    'word_replacer': {
      'configSpec': {
        'Substitutions': {
          '__class__': 'LitType',
          '__name__': 'TextSegment',
          '__mro__': ['TextSegment', 'LitType', 'object'],
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
  'littypes': {
    'Scalar': {
      '__class__': 'type',
      '__mro__': ['Scalar', 'LitType', 'object'],
      '__name__': 'Scalar'
    }
  },
  'demoMode': false,
  'defaultLayout': 'default',
  'canonicalURL': undefined
};
