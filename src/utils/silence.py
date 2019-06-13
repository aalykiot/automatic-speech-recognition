import math


def check_frame(frame, threshold):
    flag = False

    for value in frame:
        if (abs(value) > threshold):
            flag = True
            break
    return flag


def remove(signal, threshold):
    output = []

    frame_size = 2000
    frames = math.ceil(len(signal) / frame_size)

    prev = False
    for i in range(0, int(frames)):
        current_frame = signal[frame_size * i: frame_size * (i + 1)]
        flag = check_frame(current_frame, threshold)

        if (flag):
            output.extend(current_frame)
            prev = True
        else:
            next_frame = signal[frame_size * (i + 1): frame_size * (i + 2)]
            nxt = check_frame(next_frame, threshold)

            if nxt == True and prev == True:
                output.extend(current_frame)
                prev = True
            else:
                output.extend(frame_size * [0])
                prev = False

    return output
