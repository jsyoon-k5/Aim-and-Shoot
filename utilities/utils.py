import os, pickle, time
from datetime import datetime

def path_to_cousin(cousinname, filename):
    return '/'.join(
        (
            *os.path.abspath(__file__).split("\\")[:-2],
            cousinname,
            filename
        )
    )


def now_to_string():
    ct = datetime.now()
    return f"{ct.year%100:02d}{ct.month:02d}{ct.day:02d}_{ct.hour:02d}{ct.minute:02d}{ct.second:02d}_{ct.microsecond//10000:02d}"


def list2str(s, sep=','):
    return sep.join(list(map(str, s)))


def pickle_save(file, data):
    done = False
    while not done:
        try:
            with open(file, "wb") as fp:
                pickle.dump(data, fp)
            done = True
        except:
            time.sleep(0.5)
            continue


def pickle_load(file):
    with open(file, "rb") as fp:
        data = pickle.load(fp)
    return data


if __name__ == "__main__":
    os.makedirs(path_to_cousin("hello", "1/2"), exist_ok=True)