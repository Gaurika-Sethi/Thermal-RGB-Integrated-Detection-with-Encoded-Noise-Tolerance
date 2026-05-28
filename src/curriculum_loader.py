import pandas as pd


class CurriculumLoader:
    """
    Curriculum learning loader.

    Stage 1:
        clean samples

    Stage 2:
        mild corruption

    Stage 3:
        heavy corruption
    """

    def __init__(self, csv_path):

        self.df = pd.read_csv(csv_path)

    def get_stage(self, stage=1):

        if stage == 1:

            return self.df[
                self.df["severity"] <= 0.1
            ]

        elif stage == 2:

            return self.df[
                (self.df["severity"] > 0.1) &
                (self.df["severity"] <= 0.3)
            ]

        elif stage == 3:

            return self.df[
                self.df["severity"] > 0.3
            ]

        else:

            raise ValueError(
                f"Unknown stage: {stage}"
            )