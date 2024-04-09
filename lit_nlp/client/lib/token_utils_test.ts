/**
 * Testing for token_utils.ts
 */

import 'jasmine';

import * as tokenUtils from './token_utils';

describe('cleanSpmText test', () => {
  it('cleans magic underscores from SPM output', () => {
    const text = 'Summarize▁this▁sentence:\n\nOnce▁upon▁a▁time';
    expect(tokenUtils.cleanSpmText(text))
        .toEqual('Summarize this sentence:\n\nOnce upon a time');
  });
});

describe('groupTokensByRegexPrefix test', () => {
  [{
    testcaseName: 'groups tokens by word',
    tokens: ['Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':'],
    regex: /[▁\s]+/g,
    expectedGroups: [['Sum', 'mar', 'ize'], ['▁this'], ['▁sent', 'ence', ':']],
  },
   {
     testcaseName: 'groups tokens by word, handling newlines',
     tokens: [
       'Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':', '\n', '\n', 'Once',
       '▁upon', '▁a', '▁time'
     ],
     // Consecutive newlines should be their own segment.
     // Start a new word on the first non-\n afterwards.
     regex: /([▁\s]+)|(?<=\n)[^\n]/g,
     expectedGroups: [
       ['Sum', 'mar', 'ize'], ['▁this'], ['▁sent', 'ence', ':'], ['\n', '\n'],
       ['Once'], ['▁upon'], ['▁a'], ['▁time']
     ],
   },
   {
     testcaseName: 'groups tokens by sentence, simple version',
     tokens: [
       'Sent', 'ence', '▁one', '.', '▁Sent', 'ence', '▁two', '!', '▁Sent',
       'ence', '▁three', '?'
     ],
     regex: /(?<=[.?!])[▁\s]+/g,
     expectedGroups: [
       ['Sent', 'ence', '▁one', '.'],
       ['▁Sent', 'ence', '▁two', '!'],
       ['▁Sent', 'ence', '▁three', '?'],
     ],
   },
   {
     testcaseName: 'groups tokens by sentence, handling newlines',
     tokens: [
       'Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':', '\n', '\n', 'Once',
       '▁upon', '▁a', '▁time'
     ],
     // Sentence start is one of:
     // - a run of consecutive \n as its own segment
     // - any non-\n following \n
     // - whitespace or magic underscore following punctuation [.?!]
     regex: /(\n+)|((?<=\n)[^\n])|((?<=[.?!])([▁\s]+))/g,
     expectedGroups: [
       ['Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':'], ['\n', '\n'],
       ['Once', '▁upon', '▁a', '▁time']
     ],
   },
   {
     testcaseName: 'groups tokens by line',
     tokens: [
       'Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':', '\n', '\n', 'Once',
       '▁upon', '▁a', '▁time'
     ],
     // Line start is either:
     // - a run of consecutive \n as its own segment
     // - any non-\n following \n
     regex: /(\n+)|([^\n]+)/g,
     expectedGroups: [
       ['Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':'], ['\n', '\n'],
       ['Once', '▁upon', '▁a', '▁time']
     ],
   },
  ].forEach(({testcaseName, tokens, regex, expectedGroups}) => {
    it(testcaseName, () => {
      const groups = tokenUtils.groupTokensByRegexPrefix(tokens, regex);
      expect(groups).toEqual(expectedGroups);
    });
  });
});


describe('groupTokensByRegexSeparator test', () => {
  [{
    testcaseName: 'groups tokens by line',
    tokens: [
      'Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':', '\n', '\n', 'Once',
      '▁upon', '▁a', '▁time', '\n', '▁there', '▁was'
    ],
    // Line separator is one or more \n
    regex: /\n+/g,
    expectedGroups: [
      ['Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':'], ['\n', '\n'],
      ['Once', '▁upon', '▁a', '▁time'], ['\n'], ['▁there', '▁was']
    ],
  },
   {
     testcaseName: 'groups tokens by paragraph',
     tokens: [
       'Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':', '\n', '\n', 'Once',
       '▁upon', '▁a', '▁time', '\n', '▁there', '▁was'
     ],
     // Line separator is two or more \n
     regex: /\n\n+/g,
     expectedGroups: [
       ['Sum', 'mar', 'ize', '▁this', '▁sent', 'ence', ':'], ['\n', '\n'],
       ['Once', '▁upon', '▁a', '▁time', '\n', '▁there', '▁was']
     ],
   },
  ].forEach(({testcaseName, tokens, regex, expectedGroups}) => {
    it(testcaseName, () => {
      const groups = tokenUtils.groupTokensByRegexSeparator(tokens, regex);
      expect(groups).toEqual(expectedGroups);
    });
  });
});