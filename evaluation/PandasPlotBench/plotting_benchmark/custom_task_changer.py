import pandas as pd

from .task_changer import TaskChanger


class TaskShortner(TaskChanger):
    """
    Removes task style and uses shortened task description
    """

    def __init__(self, shorten_type: str, *args, **kwargs):
        data_descriptor_map = {"short": self.short_df_description}
        super().__init__(data_descriptor_map=data_descriptor_map, *args, **kwargs)
        shorten_column_map = {
            "no_style": "task__plot_description",
            "short": "_task__plot_description_short",
            "short_single": "_task__plot_description_short_single",
            "empty": "empty",
        }
        self.short_column = shorten_column_map[shorten_type]

    def style_changer(self, task_text: str, df: pd.DataFrame, dp_row: pd.Series) -> str:
        dp_row["old_task__plot_style"] = dp_row["task__plot_style"]
        return ""

    def plot_descr_changer(
        self, task_text: str, df: pd.DataFrame, dp_row: pd.Series
    ) -> str:
        dp_row["old_task__plot_description"] = dp_row["task__plot_description"]
        if self.short_column != "empty":
            return dp_row[self.short_column]
        else:
            return "Plot a given dataframe"

    def short_df_description(
        self, task_text: str, df: pd.DataFrame, dp_row: pd.Series
    ) -> str:
        descr_string = f"DataFrame with {len(df.columns)} columns and {len(df)} rows\n"
        return descr_string
