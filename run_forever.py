from person_counting import one_run

if __name__ == '__main__':
    while True:
        try:
            one_run()
        except Exception as e:
            print(str(e))
