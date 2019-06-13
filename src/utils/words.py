def split(signal):
    intervals = []
    start = 0
    end = 0
    in_word = False

    for index, value in enumerate(signal):
        if abs(value) == 0:
            if in_word:
                end = index
                intervals.append([start, end])
                in_word = False
        else:
            if in_word == False:
                start = index
                in_word = True

    sub_signals = []

    for interval in intervals:
        sub_signals.append(signal[interval[0]:interval[1]])

    return sub_signals
