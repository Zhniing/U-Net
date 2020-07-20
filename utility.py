bar_len = 40


def progress_bar(i, n):
    p = i / n * bar_len
    a = '=' * round(p)
    b = ' ' * (bar_len - round(p))
    print('\r[%s%s]' % (a, b), end='')  # 不换行
