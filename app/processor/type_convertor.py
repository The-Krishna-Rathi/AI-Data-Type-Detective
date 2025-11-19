import pandas as pd
import re

class TypeConverter:

    @staticmethod
    def apply_final_datatypes(df, final_schema):
        """
        df = original dataframe
        final_schema = { column_name: final_type }
        """
        conversion_report = {}

        for col, target_type in final_schema.items():
            report = {
                "final_type": target_type,
                "status": "success",
                "applied_steps": [],
                "errors": [],
                "nulls_before": df[col].isna().sum(),
                "nulls_after": None
            }

            try:
                # INTEGER
                if target_type == "integer":
                    df[col] = df[col].astype(int)

                # FLOAT
                elif target_type == "float":
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(r"[^0-9.\-]", "", regex=True)
                    )
                    report["applied_steps"].append("removed non-numeric characters")
                    df[col] = df[col].astype(float)

                # DATETIME
                elif target_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    report["applied_steps"].append("pd.to_datetime applied")

                # BOOLEAN
                elif target_type == "boolean":
                    df[col] = df[col].astype(str).str.lower()
                    df[col] = df[col].map({"yes": True, "no": False})
                    report["applied_steps"].append("mapped yes/no to True/False")

                # CATEGORY
                elif target_type == "category":
                    df[col] = df[col].astype("category")

                # STRING
                else:
                    df[col] = df[col].astype(str)

            except Exception as e:
                report["status"] = "failed"
                report["errors"].append(str(e))

            report["nulls_after"] = df[col].isna().sum()
            conversion_report[col] = report

        return df, conversion_report
