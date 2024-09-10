import enum

class LlmHTTPEndpoints(enum.Enum):
  GENERATE = 'predict'
  SALIENCE = 'salience'
  TOKENIZE = 'tokenize'