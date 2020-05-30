import csv

WANTED_DOSES_COUNT = 8
RELATIVE_DOSE_TIME = True


def doses_count(row):
    return sum([1 for x in row[1:-1:2] if x > 0])


def change_to_relative(row):
    i = 1  # first dose_time index
    last_time = 0
    while i < len(row) - 1:
        new_time = row[i] - last_time
        last_time = row[i]
        row[i] = new_time
        i += 2
    return row

if __name__ == '__main__':
    rows = []
    header_row = []
    with open("200k_uniform/base_dataset1_200.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header_row = next(reader)
        if WANTED_DOSES_COUNT:
            header_row = [header_row[0]] + header_row[41-2*WANTED_DOSES_COUNT:]
        print(header_row)
        for row in reader:
            row = list(map(float, row))
            if RELATIVE_DOSE_TIME:
                row = change_to_relative(row)
            if not WANTED_DOSES_COUNT:
                rows.append(row)
            elif WANTED_DOSES_COUNT == doses_count(row):
                rows.append([int(row[0])] + row[41-2*WANTED_DOSES_COUNT:])

    with open("8_doses/dataset_relative.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header_row)
        writer.writerows(rows)
