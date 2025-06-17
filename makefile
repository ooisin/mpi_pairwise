CC = mpicc
CFLAGS = -Wall -Wextra -std=c99 -g
SRCDIR = src
TARGET = pairwise

$(TARGET): $(SRCDIR)/pairwise.c $(SRCDIR)/matrix.c $(SRCDIR)/validate.c
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCDIR)/pairwise.c $(SRCDIR)/matrix.c $(SRCDIR)/validate.c

clean:
	rm -f $(TARGET) *.o $(SRCDIR)/*.o

.PHONY: clean
