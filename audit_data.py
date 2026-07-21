from data_quality import load_clean_data, print_audit_report


def main():
    _, audit_details = load_clean_data()
    print_audit_report(audit_details)


if __name__ == "__main__":
    main()
