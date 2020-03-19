from data import srdata

class BSDWED2(srdata.SRData):
    def __init__(self, args, name='BSDWED2', train=True, benchmark=True):
        data_range = [r.split('-') for r in args.data_range_bsdwed.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(BSDWED2, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self,half_mode='full'):
        names_hr, names_lr = super(BSDWED2, self)._scan(half_mode='full')
        names_hr = names_hr[self.begin - 1:self.end]
        total_num = len(names_hr)
        if half_mode == 'head':
            names_hr = names_hr[0:(total_num // 2)]
        elif half_mode == 'tail':
            names_hr = names_hr[(total_num // 2):total_num]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        if half_mode == 'head':
            names_lr = [n[0:(total_num // 2)] for n in names_lr]
        elif half_mode == 'tail':
            names_lr = [n[(total_num // 2):total_num] for n in names_lr]
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(BSDWED2, self)._set_filesystem(dir_data)
        if self.input_large: self.dir_lr += 'L'
