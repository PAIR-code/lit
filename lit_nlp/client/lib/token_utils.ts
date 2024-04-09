/**
 * @fileoverview Utils for working with tokenized text.
 */

/**
 * Evil underscore used by sentencepiece to replace spaces.
 */
export const SPM_SPACE_SENTINEL = '▁';

/**
 * Clean SPM text to make it more human-readable.
 */
export function cleanSpmText(text: string): string {
  return text.replaceAll(SPM_SPACE_SENTINEL, ' ');
}

/**
 * Use a regex to match segment prefixes. The prefix and anything
 * following it (until the next match) are treated as one segment.
 *
 * @param tokens tokens to group
 * @param matcher regex to group by; must have /g set
 * @param breakOnMatchEnd if true, will also break segments on the /end/ of
 *   a matching span in addition to the beginning.
 */
function groupTokensByRegex(
    tokens: string[], matcher: RegExp, breakOnMatchEnd: boolean): string[][] {
  const text = tokens.join('');
  const matchIdxs: Array<number|undefined> = [];
  for (const match of text.matchAll(matcher)) {
    matchIdxs.push(match.index);
    if (match.index !== undefined && breakOnMatchEnd) {
      matchIdxs.push(match.index + match[0].length);
    }
  }

  let textCharOffset = 0;  // chars into text
  let matchIdx = 0;        // indices into matches
  const groups: string[][] = [];
  let acc: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nextMatch = matchIdxs[matchIdx];

    // Look ahead to see if this token intrudes on a match.
    // If so, start a new segment before pushing the token.
    if (nextMatch !== undefined && textCharOffset + token.length > nextMatch) {
      // Don't push an empty group if the first token is part of a match.
      if (acc.length > 0 || groups.length > 0) groups.push(acc);
      acc = [];
      matchIdx += 1;
    }

    // Push the token.
    acc.push(token);
    textCharOffset += token.length;
  }
  // Finally, push any open group.
  if (acc.length > 0) groups.push(acc);
  return groups;
}

/**
 * Use a regex to match segment prefixes. The prefix and anything
 * following it (until the next match) are treated as one segment.
 * For example, groupTokensByRegexPrefix(tokens, /Example:/g) will
 * create a segment each time the text "Example:" is seen.
 */
export function groupTokensByRegexPrefix(tokens: string[], matcher: RegExp) {
  return groupTokensByRegex(tokens, matcher, /* breakOnMatchEnd */ false);
}

/**
 * Use a regex to match a separator segment. A matching span is treated
 * as a segment, and anything between matches is treated as a separate segment.
 * For example, groupTokensByRegexSeparator(tokens, /\n+/g) will group tokens
 * in between newlines, with any sequence of \n as its own segment.
 */
export function groupTokensByRegexSeparator(tokens: string[], matcher: RegExp) {
  return groupTokensByRegex(tokens, matcher, /* breakOnMatchEnd */ true);
}