CC = nvcc
CFLAGS = -O3 -Xcompiler -Wall

THRUST = thrust
SINGLETHREAD = singlethread
MULTITHREAD = multithread

make: thrust singlethread multithread

thrust:
	$(CC) $(CFLAGS) $(THRUST).cu -o $(THRUST)

singlethread:
	$(CC) $(CFLAGS) $(SINGLETHREAD).cu -o $(SINGLETHREAD)

multithread:
	$(CC) $(CFLAGS) $(MULTITHREAD).cu -o $(MULTITHREAD)

clean:
	rm $(THRUST) $(SINGLETHREAD) $(MULTITHREAD)