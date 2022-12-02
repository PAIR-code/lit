import 'jasmine';
import {MulticlassPreds} from '../lib/lit_types';
import {getMarginFromThreshold} from '../lib/utils';
import {GroupedExamples, ModelSpec} from '../lib/types';
import {AppState} from './state_service';
import {ClassificationService} from './classification_service';

const FIELD_NAME = 'pred';
const MODEL_NAME = 'test_model';

const MULTICLASS_PRED_WITH_THRESHOLD = new MulticlassPreds();
MULTICLASS_PRED_WITH_THRESHOLD.null_idx = 0;
MULTICLASS_PRED_WITH_THRESHOLD.vocab = ['0', '1'];
MULTICLASS_PRED_WITH_THRESHOLD.threshold = 0.3;
const MULTICLASS_SPEC_WITH_THRESHOLD: ModelSpec = {
  input: {},
  output: {[FIELD_NAME]: MULTICLASS_PRED_WITH_THRESHOLD}
};

const MULTICLASS_PRED_WITHOUT_THRESHOLD = new MulticlassPreds();
MULTICLASS_PRED_WITHOUT_THRESHOLD.null_idx = 0;
MULTICLASS_PRED_WITHOUT_THRESHOLD.vocab = ['0', '1'];
const MULTICLASS_SPEC_WITHOUT_THRESHOLD: ModelSpec = {
  input: {},
  output: {[FIELD_NAME]: MULTICLASS_PRED_WITHOUT_THRESHOLD}
};

const MULTICLASS_PRED_NO_VOCAB = new MulticlassPreds();
MULTICLASS_PRED_NO_VOCAB.null_idx = 0;
const INVALID_SPEC_NO_VOCAB: ModelSpec = {
  input: {},
  output: {[FIELD_NAME]: MULTICLASS_PRED_NO_VOCAB}
};

const MULTICLASS_PRED_NO_NULL_IDX = new MulticlassPreds();
MULTICLASS_PRED_NO_NULL_IDX.vocab = ['0', '1'];
const INVALID_SPEC_NO_NULL_IDX: ModelSpec = {
  input: {},
  output: {[FIELD_NAME]: MULTICLASS_PRED_NO_NULL_IDX}
};

const INVALID_SPEC_NO_MULTICLASS_PRED: ModelSpec = {
  input: {},
  output: {}
};

const UPDATED_MARGIN = getMarginFromThreshold(0.8);

type MinimalAppState = Pick<AppState, 'currentModels' | 'currentModelSpecs'>;

describe('classification service test', () => {
  [   // Parameterized tests for models with valid specs.
    {
      name: 'without a threshold',
      spec: MULTICLASS_SPEC_WITHOUT_THRESHOLD,
      facets: undefined,
      expThreshold: undefined,
      expMargin: 0
    },
    {
      name: 'without a threshold with facets',
      spec: MULTICLASS_SPEC_WITHOUT_THRESHOLD,
      facets: ['TN', 'TP'],
      expThreshold: undefined,
      expMargin: 0
    },
    {
      name: 'with a threshold',
      spec: MULTICLASS_SPEC_WITH_THRESHOLD,
      facets: undefined,
      expThreshold: 0.3,
      expMargin: getMarginFromThreshold(0.3)
    },
    {
      name: 'with a threshold and facets',
      spec: MULTICLASS_SPEC_WITH_THRESHOLD,
      facets: ['TN', 'TP'],
      expThreshold: 0.3,
      expMargin: getMarginFromThreshold(0.3)
    },
  ].forEach(({name, spec, facets, expThreshold, expMargin}) => {
    const mockAppState: MinimalAppState = {
      currentModels: [MODEL_NAME],
      currentModelSpecs: {[MODEL_NAME]: {
        spec,
        datasets: [],
        generators: [],
        interpreters: []
      }}
    };

    const classificationService =
        new ClassificationService(mockAppState as {} as AppState);

    function getMargin (facet?: string) {
      const facetData = facet != null ?
          {displayName: facet, data: [], facets: {}} : undefined;
      return classificationService.getMargin(MODEL_NAME, FIELD_NAME, facetData);
    }

    it(`derives margin settings from a spec ${name}`, () => {
      const predSpec = spec.output['pred'];
      expect(predSpec).toBeInstanceOf(MulticlassPreds);
      expect((predSpec as MulticlassPreds).threshold).toEqual(expThreshold);
      expect(getMargin()).toBe(expMargin);
    });

    it(`updates margin settings for specs ${name}`, () => {
      classificationService.setMargin(MODEL_NAME, FIELD_NAME, UPDATED_MARGIN);
      expect(getMargin()).toBe(UPDATED_MARGIN);
    });

    it(`resets margin settings for specs ${name}`, () => {
      classificationService.resetMargins({[MODEL_NAME]: spec.output});
      expect(getMargin()).toBe(expMargin);
    });

    if (facets != null) {
      const groupedExamples = facets.reduce((obj, facet) => {
        obj[facet] = {data: [], displayName: facet, facets: {}};
        return obj;
      }, {} as GroupedExamples);

      classificationService.setMarginGroups(MODEL_NAME, FIELD_NAME,
                                            groupedExamples);

      for (const facet of facets) {
        it(`derives margin for ${facet} facet from a spec ${name}`, () => {
            expect(getMargin(facet)).toBe(expMargin);
        });

        it(`updates margin for ${facet} facet from a spec ${name}`, () => {
          const facetData = {displayName: facet, data: [], facets: {}};
          classificationService.setMargin(MODEL_NAME, FIELD_NAME,
                                          UPDATED_MARGIN, facetData);
          expect(getMargin(facet)).toBe(UPDATED_MARGIN);
        });

        it(`resets margin for ${facet} facet from a spec ${name}`, () => {
          classificationService.resetMargins({[MODEL_NAME]: spec.output});
          expect(getMargin(facet)).toBe(expMargin);
        });
      }
    }
  });

  [   // Parameterized tests for models with invalid specs
    {
      name: 'without a multiclass pred',
      spec: INVALID_SPEC_NO_MULTICLASS_PRED,
    },
    {
      name: 'without null_idx',
      spec: INVALID_SPEC_NO_NULL_IDX,
    },
    {
      name: 'without vocab',
      spec: INVALID_SPEC_NO_VOCAB,
    },
  ].forEach(({name, spec}) => {
    it(`should not compute margins ${name}`, () => {
      const mockAppState: MinimalAppState = {
        currentModels: [MODEL_NAME],
        currentModelSpecs: {[MODEL_NAME]: {
          spec,
          datasets: [],
          generators: [],
          interpreters: []
        }}
      };

      const {marginSettings} =
          new ClassificationService(mockAppState as {} as AppState);

      expect(marginSettings[MODEL_NAME]).toBeDefined();
      expect(marginSettings[MODEL_NAME][FIELD_NAME]).toBeUndefined();
    });
  });
});
