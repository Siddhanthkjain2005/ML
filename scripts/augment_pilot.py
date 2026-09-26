from pathlib import Path
from ber.augment import augment


def main():
 for fold in ('train','calibration'):
  root=Path('experiments/pilot_v1')/fold
  augment(root/'candidates',root/'features','work/vectorizers_romanized_v1.joblib',root/'features_advanced_v1')

if __name__=='__main__':main()
