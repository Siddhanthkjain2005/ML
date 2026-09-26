from pathlib import Path
from ber.reference_context import augment_reference


def main():
 for fold in ('train','calibration'):
  root=Path('experiments/pilot_v1')/fold
  augment_reference(root/'candidates',root/'features_advanced_v1',root/'features',Path('work/splits_v1')/fold/'source1.tsv',root/'features_reference_v1',Path('work/reference_indices')/fold)

if __name__=='__main__':main()
