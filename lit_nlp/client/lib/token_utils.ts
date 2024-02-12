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
 */
export function groupTokensByRegexPrefix(
    tokens: string[],
    matcher: RegExp,
    ): string[][] {
  const text = tokens.join('');
  const matches = [...text.matchAll(matcher)];

  let textCharOffset = 0;  // chars into text
  let matchIdx = 0;        // indices into matches
  const groups: string[][] = [];
  let acc: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    const nextMatch = matches[matchIdx];

    // Look ahead to see if this token intrudes on a match.
    // If so, start a new segment before pushing the token.
    if (nextMatch !== undefined &&
        textCharOffset + token.length > nextMatch.index!) {
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