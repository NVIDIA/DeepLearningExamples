from pytablewriter import MarkdownTableWriter


class TrainingTable:
    def __init__(self, acc_unit='BLEU', time_unit='min', perf_unit='tok/s'):
        self.data = []
        self.acc_unit = acc_unit
        self.time_unit = time_unit
        self.perf_unit = perf_unit
        self.time_unit_convert = {'s': 1, 'min': 1/60, 'h': 1/3600}

    def add(self, gpus, batch_size, accuracy, perf, time_to_train):
        time_to_train *= self.time_unit_convert[self.time_unit]
        if not accuracy:
            accuracy = 0.0
        accuracy = round(accuracy, 2)
        self.data.append([gpus, batch_size, accuracy, perf, time_to_train])

    def write(self, title, math):
        writer = MarkdownTableWriter()
        writer.table_name = f'{title}'

        header = [f'**GPUs**',
                  f'**Batch Size / GPU**',
                  f'**Accuracy - {math.upper()} ({self.acc_unit})**',
                  f'**Throughput - {math.upper()} ({self.perf_unit})**',
                  f'**Time to Train - {math.upper()} ({self.time_unit})**',
                  ]
        writer.headers = header

        writer.value_matrix = self.data
        writer.write_table()
