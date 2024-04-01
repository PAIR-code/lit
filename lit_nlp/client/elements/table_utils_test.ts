import 'jasmine';

import {TableEntry, TableRowInternal} from './table_types';
import * as tableUtils from './table_utils';

const REGEX_QUERY = '^test.*ng';
const COLUMN_NAMES = [
  'colA',
  'colB',
  'colC',
  't2tgenai:response',
  'multiclass:clspreds:class'
];

const JoinedQueryType = tableUtils.JoinedQueryType;
type JoinedQueryTestSpec = [
  testName: string,
  searchText: string,
  expected: tableUtils.JoinedQuery | null
];
type SearchQueryTestSpec = [
  testName: string,
  searchText: string,
  expected: tableUtils.SearchQuery
];

function searchQuery(
    matcher: string, columnName?: string): tableUtils.SearchQuery {
  return columnName ? {columnName, matcher} : {matcher};
}

function searchExpr(searchText: string, columnName?: string) {
  return {
    type: JoinedQueryType.EXPR,
    children: searchQuery(searchText, columnName)
  };
}


describe('makeSearchQuery', () => {
  const columnNames = [
    'colA', 'colB', 'colC', 'model:field', 'model:field:derived'
  ];
  const testCases: SearchQueryTestSpec[] = [
    [
      'any column text search',
      'foo',
      searchQuery('foo')
    ],
    [
      'any column numeric search',
      '1',
      searchQuery('1')
    ],
    [
      'specific column regex wildcard',
      'colA:foo.*',
      searchQuery('foo.*', 'colA')
    ],
    [
      'specific column position regex',
      'colB:^.*',
      searchQuery('^.*', 'colB')
    ],
    [
      'specific column numeric search',
      'colX:123',
      searchQuery('colX:123')
    ],
    [
      'missing column coerced to any column text search',
      ':123',
      searchQuery(':123')
    ],
    [
      'model column text search',
      'model:field:123',
      searchQuery('123', 'model:field')
    ],
    [
      'derived column numeric search',
      'model:field:derived:1',
      searchQuery('1', 'model:field:derived')
    ],
  ];
  for (const [testName, searchText, expected] of testCases) {
    it(`makes search queries for ${testName}`, () => {
      expect(tableUtils.makeSearchQuery(searchText, columnNames))
          .toEqual(expected);
    });
  }
});

describe('parseSearchTextIntoQueries', () => {
  const query1 = searchExpr('foo', 'colA');
  const query2 = searchExpr('bar');
  const query3 = searchExpr('123', 'colB');
  const query4 = searchExpr('f.*');
  const modelColumnQuery = searchExpr('bar', 't2tgenai:response');
  const derivedModelColumnQuery =
      searchExpr('bar', 'multiclass:clspreds:class');

  const testCases: JoinedQueryTestSpec[] = [
    [
      'a single query on any field',
      'foo bar',
      searchExpr('foo bar')
    ],
    [
      'a single query on model field',
      't2tgenai:response:bar',
      modelColumnQuery
    ],
    [
      'a single query on derived field for a model prediction',
      'multiclass:clspreds:class:bar',
      derivedModelColumnQuery
    ],
    [
      'AND-separated dataset column and any column queries',
      'colA:foo AND bar',
      {type: JoinedQueryType.AND, children: [query1, query2]}
    ],
    [
      'AND-separated model column queries',
      't2tgenai:response:bar AND multiclass:clspreds:class:bar',
      {
        type: JoinedQueryType.AND,
        children: [modelColumnQuery, derivedModelColumnQuery]
      }
    ],
    [
      'OR-separated model column queries',
      't2tgenai:response:bar OR multiclass:clspreds:class:bar',
      {
        type: JoinedQueryType.OR,
        children: [modelColumnQuery, derivedModelColumnQuery]
      }
    ],
    [
      'Many OR-separated queries',
      'colA:foo OR bar OR f.*',
      {type: JoinedQueryType.OR, children: [query1, query2, query4]}
    ],
    [
      'joint AND-OR queries with a single OR',
      'colA:foo AND bar OR f.*', {
        type: JoinedQueryType.AND,
        children:
            [query1, {type: JoinedQueryType.OR, children: [query2, query4]}]
      }
    ],
    [
      'joint AND-OR queries with a OR-queries on either side',
      'colB:123 OR bar AND colA:foo OR f.*', {
        type: JoinedQueryType.AND,
        children: [
          {type: JoinedQueryType.OR, children: [query3, query2]},
          {type: JoinedQueryType.OR, children: [query1, query4]},
        ]
      }
    ],  // Joint queries.
    [
      'joint AND-OR queries with two AND clauses and a single OR clause',
      'colA:foo AND f.* AND colB:123 OR bar', {
        type: JoinedQueryType.AND,
        children: [
          query1,
          query4,
          {
            type: JoinedQueryType.OR,
            children: [query3, query2]
          }
        ]
      }
    ]
  ];
  for (const [testName, searchText, expected] of testCases) {
    it(`correctly parses search text for ${testName}`, () => {
      expect(tableUtils.parseSearchTextIntoQueries(searchText, COLUMN_NAMES))
          .toEqual(expected);
    });
  }

  const cornerTestCases: JoinedQueryTestSpec[] = [
    [
      'empty string', '', null
    ],
    [
      'only the AND token', ' AND ', null
    ],
    [
      'only the OR token', ' OR ', null
    ],
    [
      'sequential AND tokens', ' AND AND', searchExpr('AND')
    ],
    [
      'an AND token followed by a query',
      ' AND colA:foo',
      searchExpr('foo', 'colA')
    ],
    [
      'an AND with an OR on the RHS where the RHS of the OR is empty',
      'f.* AND bar OR ',
      {type: JoinedQueryType.AND, children: [query4, query2]}
    ],
    [
      'an AND where the LHS is empty and the RHS of the final OR is empty',
      ' AND colB:123 OR bar OR ',
      {type: JoinedQueryType.OR, children: [query3, query2]}
    ],
    [
      'an invalid AND on the LHS of an OR',
      'AND 100 OR bar ',
      {
        type: JoinedQueryType.OR,
        children: [
          searchExpr('AND 100'),
          searchExpr('bar '),
        ]
      }
    ],
  ];

  for (const [testName, searchText, expected] of cornerTestCases) {
    it(`correctly parses search text for ${testName}`, () => {
      expect(tableUtils.parseSearchTextIntoQueries(searchText, COLUMN_NAMES))
          .toEqual(expected);
    });
  }
});

describe('itemMatchesText', () => {
  it('does not filter images', () => {
    const imageItem = 'data:image/png:test';
    const nonImageItem = 'not-an-image/png:test';
    const query = 'foobar';

    expect(tableUtils.itemMatchesText(imageItem, query)).toBe(true);
    expect(tableUtils.itemMatchesText(nonImageItem, query)).toBe(false);
  });

  it('correctly matches regex queries', () => {
    const items = ['test*ng', 'testiiing'];
    for (const item of items) {
      expect(tableUtils.itemMatchesText(item, REGEX_QUERY)).toBe(true);
    }
  });

  it('correctly does not match regex queries', () => {
    const items = ['atestiiing', 'atestng'];
    for (const item of items) {
      expect(tableUtils.itemMatchesText(item, REGEX_QUERY)).toBe(false);
    }
  });

  it('handles numerical searching', () => {
    const rangeQuery = '1-10';
    expect(tableUtils.itemMatchesText('9', rangeQuery)).toBe(true);

    const numberQuery = '8';
    expect(tableUtils.itemMatchesText(9, numberQuery)).toBe(false);

    const regexNumberQuery = '^8';
    expect(tableUtils.itemMatchesText(8, regexNumberQuery)).toBe(true);
    expect(tableUtils.itemMatchesText(18, regexNumberQuery)).toBe(false);
  });
});

function makeTableRow(entries: TableEntry[]): TableRowInternal {
  return {inputIndex: 1, rowData: entries};
}

const TEST_ROW_DATA = ['foo', 'bar', '123'];
const TEST_ROW_DATA2 = ['fizz', 'buzz', '456'];
const TEST_ROW_DATA3 = ['red', 'sox', '500'];

describe('tableRowMatchesSingleQuery', () => {
  const matchCases = ['f', '^f', '100-200'];
  for (const searchText of matchCases) {
    it('correctly matches table rows for singular text queries', () => {
      const query = searchQuery(searchText);
      expect(tableUtils.tableRowMatchesSingleQuery(
                 makeTableRow(TEST_ROW_DATA), COLUMN_NAMES, query))
          .toBe(true);
    });
  }

  const noMatchCases = ['t', 'x$', '-20--10'];
  for (const searchText of noMatchCases) {
    it('correctly does not match table rows for singular text queries', () => {
      const query = searchQuery(searchText);
      expect(tableUtils.tableRowMatchesSingleQuery(
                 makeTableRow(TEST_ROW_DATA), COLUMN_NAMES, query))
          .toBe(false);
    });
  }

  const columnCases = [
    searchQuery('foo', 'colB'), searchQuery('b.*', 'colA'),
    searchQuery('100-200', 'colA')
  ];
  for (const query of columnCases) {
    it('correctly does not match table rows with column specifications', () => {
      expect(tableUtils.tableRowMatchesSingleQuery(
                 makeTableRow(TEST_ROW_DATA), COLUMN_NAMES, query))
          .toBe(false);
    });
  }
});

describe('filterDataByQueries', () => {
  const tableRows = [
    makeTableRow(TEST_ROW_DATA), makeTableRow(TEST_ROW_DATA2),
    makeTableRow(TEST_ROW_DATA3)
  ];

  const query1 = searchExpr('^z');
  const query2 = searchExpr('600-700');
  const query3 = searchExpr('800-900');
  const query4 = searchExpr('fooey');

  const noMatchCases = [
    {type: JoinedQueryType.AND, children: [query1, query2, query3]},
    {
      type: JoinedQueryType.AND,
      children: [
        query1, {type: JoinedQueryType.OR, children: [query2, query3]}, query4
      ]
    },
    searchExpr('-100--90'),
    searchExpr('100--100'),
  ];
  for (const queries of noMatchCases) {
    it('correctly does not match any data', () => {
      const result =
          tableUtils.filterDataByQueries(tableRows, COLUMN_NAMES, queries);
      expect(result.map(r => r.rowData)).toEqual([]);
    });
  }

  const query5 = searchExpr('fizz');
  const query6 = searchExpr('z*');
  const query7 = searchExpr('b.*');
  const query8 = searchExpr('456');

  const oneMatchCases = [
    {type: JoinedQueryType.AND, children: [query5, query8]}, {
      type: JoinedQueryType.AND,
      children: [
        searchExpr('400-500'), query5,
        {type: JoinedQueryType.OR, children: [query6, query7]}
      ]
    },
    {
      type: JoinedQueryType.AND,
      children: [
        searchExpr('fizz', 'colA'),
        searchExpr('4.*6'),
        {
          type: JoinedQueryType.OR,
          children: [query6, searchExpr('b.*', 'colB')]
        }
      ]
    }
  ];

  for (const query of oneMatchCases) {
    it('correctly matches one row', () => {
      const result =
          tableUtils.filterDataByQueries(tableRows, COLUMN_NAMES, query);
      expect(result.map(r => r.rowData)).toEqual([TEST_ROW_DATA2]);
    });
  }

  const twoMatchCases = [
    searchExpr('400-501'),
    {
      type: JoinedQueryType.OR,
      children: [searchExpr('500-550'), searchExpr('400-480')]
    },
    {
      type: JoinedQueryType.AND,
      children: [
        {
          type: JoinedQueryType.OR,
          children: [searchExpr('red'), searchExpr('buzz', 'colB')]
        },
        searchExpr('400-600')
      ]
    }
  ];
  for (const query of twoMatchCases) {
    it('correctly matches two rows', () => {
      const result =
          tableUtils.filterDataByQueries(tableRows, COLUMN_NAMES, query);
      expect(result.map(r => r.rowData)).toEqual([
        TEST_ROW_DATA2, TEST_ROW_DATA3
      ]);
    });
  }

  const allMatchCases = [searchExpr('.*'), searchExpr('100-1000')];
  for (const query of allMatchCases) {
    it('correctly matches all rows', () => {
      const result =
          tableUtils.filterDataByQueries(tableRows, COLUMN_NAMES, query);
      expect(result.map(r => r.rowData)).toEqual([
        TEST_ROW_DATA, TEST_ROW_DATA2, TEST_ROW_DATA3
      ]);
    });
  }
});
