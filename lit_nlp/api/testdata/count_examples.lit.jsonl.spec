{
  "parity": {
    "required": true,
    "annotated": false,
    "default": "",
    "vocab": [
      "odd",
      "even"
    ],
    "__class__": "LitType",
    "__name__": "CategoryLabel"
  },
  "text": {
    "required": true,
    "annotated": false,
    "default": "",
    "__class__": "LitType",
    "__name__": "TextSegment"
  },
  "value": {
    "required": true,
    "annotated": false,
    "min_val": -32768,
    "max_val": 32767,
    "default": 0,
    "step": 1,
    "__class__": "LitType",
    "__name__": "Integer"
  },
  "other_divisors": {
    "required": true,
    "annotated": false,
    "default": [],
    "vocab": null,
    "separator": ",",
    "__class__": "LitType",
    "__name__": "SparseMultilabel"
  },
  "in_spanish": {
    "required": true,
    "annotated": false,
    "default": "",
    "__class__": "LitType",
    "__name__": "TextSegment"
  },
  "embedding": {
    "required": true,
    "annotated": false,
    "default": null,
    "__class__": "LitType",
    "__name__": "Embeddings"
  }
}