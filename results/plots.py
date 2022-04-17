from matplotlib import pyplot as plt


def plot_multiline_series(
    x: list[float],
    y: list[list[float]],
    labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    y_lim: tuple[int, int] = (0, 1.1)
) -> None:
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)

    plt.ylim(*y_lim)

    plt.plot(x, y, label=labels)
    plt.legend()
    plt.show()
