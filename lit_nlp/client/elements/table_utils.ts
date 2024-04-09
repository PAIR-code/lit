/**
 * @license
 * Copyright 2022 Google LLC
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

/**
 * Utilities used by the data table component
 */

import {isTemplateResult} from 'lit/directive-helpers.js';

import {canParseNumberRangeFnFromString, isNumber, numberRangeFnFromString} from '../lib/utils';

import {SortableTableEntry, SortableTemplateResult, TableEntry, TableRowInternal} from './table_types';

/** A regex-supported string used for searching, with optional column name. */
export interface SearchQuery {
  columnName?: string;  // If undefined, search all columns.
  matcher: string;      // A regex-supported string.
}

/** Types of joined queries. */
export enum JoinedQueryType {
  AND,  // An and- joined expression, e.g. foo AND bar
  OR,   // An or- joined expression, e.g. foo OR bar
  EXPR  // A single expression, e.g. foo
}

/** An interface used to express a joint search query. */
export interface JoinedQuery {
  type: JoinedQueryType;
  // If type is Expr, this is a SearchQuery.
  children: SearchQuery|JoinedQuery[];
}

/** Returns a sortable casting of the given table entry. */
export function getSortableEntry(colEntry: TableEntry): SortableTableEntry {
  // Passthrough values if TableEntry is number or string. If it is
  // TemplateResult return 0 for sorting purposes. If it is a sortable
  // tempate result then sort by the underlying sortable value.
  if (typeof colEntry === 'string' || isNumber(colEntry)) {
    return colEntry as SortableTableEntry;
  }
  if (isTemplateResult(colEntry)) {
    return 0;
  }
  return (colEntry as SortableTemplateResult).value;
}

/**
 * Parses the search text into a search query.
 *
 * Optional column names are used to parse `columnName:searchText`-formatted
 * queries.
 */
export function makeSearchQuery(
    searchText: string, columnNames: string[] = []): SearchQuery {
  // The searchText contains no ' AND ' or ' OR ' tokens.
  // Determine if /[\w:]+search_query/ syntax matches.
  const lastColonIndex = searchText.lastIndexOf(':');

  if (lastColonIndex === -1) {
    return {matcher: searchText};
  }

  // String contains a colon. Attempt to find the longest possible column name
  // from the elements returned by splitting on the colon, starting with the
  // penultimate element.
  const searchComponents = searchText.split(':');
  let longestValidColumnName = '';
  let searchComponentIndex = searchComponents.length - 1;
  while (searchComponentIndex >= 0) {
    const candidateColumnName =
        searchComponents.slice(0, searchComponentIndex).join(':');
    if (columnNames.indexOf(candidateColumnName) !== -1) {
      longestValidColumnName = candidateColumnName;
      break;
    }
    searchComponentIndex -= 1;
  }

  // We found a valid column name, so return the search query with the column
  // name and the remaining search text.
  if (longestValidColumnName !== '') {
    return {
      columnName: longestValidColumnName,
      matcher: searchComponents.slice(searchComponentIndex).join(':')
    };
  }

  // Otherwise, return the full search text as the matcher.
  return {matcher: searchText};
}

/** Returns a JoinedQuery instance parsed from the search text. */
export function parseSearchTextIntoQueries(
    searchText: string, columnNames: string[]): JoinedQuery|null {
  if (!searchText) return null;

  const andTokenText = ' AND ';
  const orTokenText = ' OR ';

  // Recursively parse and add queries from all sides of an ' AND '-separated
  // query.
  if (searchText.indexOf(andTokenText) !== -1) {
    const andSections = searchText.split(andTokenText);
    const children =
        andSections.map(s => parseSearchTextIntoQueries(s, columnNames))
            .filter(Boolean);

    if (children.length === 1) {
      return children[0];
    } else if (children.length > 1) {
      return {type: JoinedQueryType.AND, children} as JoinedQuery;
    }
    return null;
  }

  // Split ' OR '-separated queries.
  if (searchText.indexOf(orTokenText) !== -1) {
    // Filter to remove empty items.
    const orTokens = searchText.split(orTokenText)
                         .filter(Boolean)
                         .map(t => makeSearchQuery(t, columnNames));
    if (orTokens.length === 1) {
      return {type: JoinedQueryType.EXPR, children: orTokens[0]};
    } else if (orTokens.length > 1) {
      return {
        type: JoinedQueryType.OR,
        children: orTokens.map(t => ({type: JoinedQueryType.EXPR, children: t}))
      };
    }
    return null;
  }

  return {
    type: JoinedQueryType.EXPR,
    children: makeSearchQuery(searchText, columnNames)
  };
}

/**
 * Returns true if the item matches the search text.
 */
export function itemMatchesText(
    item: SortableTableEntry, searchText: string): boolean {
  const numericItem = Number(item);
  if (canParseNumberRangeFnFromString(searchText) && !isNaN(numericItem)) {
    const matchFn = numberRangeFnFromString(searchText);
    return matchFn(numericItem);
  }

  item = String(item);
  // Don't match images.
  if (item.startsWith('data:image/png')) {
    return true;
  } else {
    try {
      return item.match(new RegExp(searchText)) !== null;
    } catch (e) {
      // Do a simple match if the query is invalid regex.
      return item.includes(searchText);
    }
  }
}

/**
 * Returns true if the table row matches the search query.
 */
export function tableRowMatchesSingleQuery(
    item: TableRowInternal, columnNames: string[],
    searchFilter: SearchQuery): boolean {
  if (searchFilter.columnName !== undefined) {
    // Note: For a performance improvement, consider caching a dictionary of
    // table names to indices and passing through to this method as a parameter.
    // Tradeoffs discussed in cl/482874768.
    // Tracking: see b/189178146.
    const col = getSortableEntry(
        item.rowData[columnNames.indexOf(searchFilter.columnName)]);
    return itemMatchesText(col, searchFilter.matcher);
  } else {
    return item.rowData.some(
        i => itemMatchesText(getSortableEntry(i), searchFilter.matcher));
  }
}

/**
 * Returns true if the table row matches the joined query.
 */
export function tableRowMatchesJoinedQuery(
    tableRow: TableRowInternal, columnNames: string[],
    joinedQuery: JoinedQuery): boolean {
  if (joinedQuery.type === JoinedQueryType.EXPR) {
    return tableRowMatchesSingleQuery(
        tableRow, columnNames, joinedQuery.children as SearchQuery);
  } else {
    const queries = joinedQuery.children as JoinedQuery[];
    if (joinedQuery.type === JoinedQueryType.AND) {
      return queries.every(
          q => tableRowMatchesJoinedQuery(tableRow, columnNames, q));
    } else {  // This is an OR-joined query.
      return queries.some(
          q => tableRowMatchesJoinedQuery(tableRow, columnNames, q));
    }
  }
}

/**
 * Returns data filtered by a list of search queries.
 */
export function filterDataByQueries(
    data: TableRowInternal[], columnNames: string[],
    joinedQuery: JoinedQuery|null) {
  if (!joinedQuery) {
    return data;
  }

  return data.filter(
      row => tableRowMatchesJoinedQuery(row, columnNames, joinedQuery));
}
