import os

if 'NE4LP_DIR' not in os.environ:
    import warnings

    warnings.warn(f'No environment variable NE4LP_DIR found. ne4lp dir is '
                  f'assumed to be current directory which is '
                  f'{os.path.abspath(".")}. Data is assumed to be found '
                  f'in {os.path.join(os.path.abspath("."), "data")}. '
                  f'If this is not the case, specify the exact location where'
                  f'ne4lp should live as environment variable.')
else:
    print(f'location of ne4lp directory: '
          f'{os.path.abspath(os.environ["NE4LP_DIR"])}')

from .sim import SIMILARITY_INDICES
from .emb import EMBEDDING_METHODS