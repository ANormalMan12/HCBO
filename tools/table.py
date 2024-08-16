
from tabulate import tabulate
def show_pd_table(dataf,file=None):
    table = tabulate(dataf,headers='keys',tablefmt='fancy_grid')
    if(file is not None):
        print(table,file=file)
    else:
        print(table,file)