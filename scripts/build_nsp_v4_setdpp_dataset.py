# This builder intentionally reuses the NSP-v3 listwise dataset format.
# SetDPP training consumes grouped rows from the same schema.

from scripts.build_nsp_v3_listwise_dataset import main


if __name__ == "__main__":
    main()
