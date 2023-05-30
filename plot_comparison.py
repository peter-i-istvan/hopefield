import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv('results.csv')
    name = df['test_name']
    h = df['hopefield_time']
    i = df['ip_time']
    plt.plot(name, h, label='Hopefield')
    plt.plot(name, i, label='IP solver')
    plt.xticks(rotation=45)
    plt.ylabel('Time [s]')
    plt.xlabel('Test case')
    plt.legend()
    plt.title('Runtime comparison')
    plt.savefig('time.png')
    plt.clf()
    h = df['hopefield_optimum']
    i = df['ip_optimum']
    c = [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 15, 15, 20, 20, 40, 40]
    plt.scatter(c, h, label='Hopefield')
    plt.scatter(c, i, label='IP')
    plt.legend()
    plt.xlabel('Number of cities')
    plt.ylabel('Predicted optimal sum of distance')
    plt.savefig('OPT.png')


if __name__ == '__main__':
    main()