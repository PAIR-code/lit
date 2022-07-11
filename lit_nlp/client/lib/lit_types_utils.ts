import {Spec} from '../lib/types';

import {LitName, LitType, REGISTRY} from './lit_types';

/**
 * Creates and returns a new LitType instance.
 * @param typeName: The name of the desired LitType.
 * @param constructorParams: A dictionary of properties to set on the LitType.
 * For example, {'show_in_data_table': true}.
 */
export function createLitType(
    typeName: LitName, constructorParams: {[key: string]: unknown} = {}) {
  const litType = REGISTRY[typeName];

  // tslint:disable-next-line:no-any
  const newType = new (litType as any)();
  newType.__name__ = typeName;
  newType.__mro__ = getMethodResolutionOrder(newType);

  for (const key in constructorParams) {
    if (key in newType) {
      newType[key] = constructorParams[key];
    } else {
      throw new Error(
          `Attempted to set unrecognized property ${key} on ${newType}.`);
    }
  }

  return newType;
}

/**
 * Returns the method resolution order for a given litType.
 * This is for compatability with references to non-class-based LitTypes,
 * and should match the Python class hierarchy.
 */
export function getMethodResolutionOrder(litType: LitType): string[] {
  const mro: string[] = [];

  // TODO(b/162269499): Remove this method after we replace the old LitType.
  let object = Object.getPrototypeOf(litType);
  while (object) {
    mro.push(object.constructor.name);
    object = Object.getPrototypeOf(object);
  }

  return mro;
}

/**
 * Returns whether the litType is a subtype of any of the typesToFind.
 * @param litType: The LitType to check.
 * @param typesToFind: Either a single or list of parent LitType candidates.
 */
export function isLitSubtype(litType: LitType, typesToFind: LitName|LitName[]) {
  if (litType == null) return false;

  if (typeof typesToFind === 'string') {
    typesToFind = [typesToFind];
  }

  for (const typeName of typesToFind) {
    // tslint:disable-next-line:no-any
    const registryType : any = REGISTRY[typeName];

    if (litType instanceof registryType) {
      return true;
    }
  }
  return false;
}

/**
 * Returns all keys in the given spec that are subtypes of the typesToFind.
 * @param spec: A Spec object.
 * @param typesToFind: Either a single or list of parent LitType candidates.
 */
export function findSpecKeys(
    spec: Spec, typesToFind: LitName|LitName[]): string[] {
  return Object.keys(spec).filter(
      key => isLitSubtype(spec[key], typesToFind));
}

/**
 * Try to cast the unknown litType as any of the candidate typesToTry.
 * Returns the first appropriate cast in the list, or null otherwise.
 */
export function tryCastAsType(
    litType: unknown, typesToTry: LitName|LitName[], throwErrorIfFail = false) {
  if (typeof typesToTry === 'string') {
    typesToTry = [typesToTry];
  }

  for (const typeName of typesToTry) {
    // tslint:disable-next-line:no-any
    const registryType: any = REGISTRY[typeName];

    if (litType instanceof registryType) {
      return litType as typeof registryType;
    }
  }

  if (throwErrorIfFail) {
    throw new TypeError(`Unable to cast type as ${typesToTry}: ${litType}`);
  }

  return null;
}
