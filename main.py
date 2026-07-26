import pandas as pd

def main():
    colspecs = [
        (0, 5),     # order number
        (6, 13),    # ICD-10-CM code
        (14, 15),   # description type (0 = category header, 1 = billable code)
        (16, 76),   # short description (60 chars)
        (77, None), # long description (variable)
    ]
    names = ["order", "code", "type", "short_desc", "long_desc"]

    df = pd.read_fwf(
        "icd10cm_order_2026.txt",
        colspecs=colspecs,
        names=names,
        dtype={"order": "int64", "code": "string", "type": "int8"},
    )

    print(df.head())
    


if __name__ == "__main__":
    main()
