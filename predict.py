from evaluate_scuia import main
import sys
import traceback


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
