CC=nvc
CFLAGS?=-Wall -tp=native -c17 -mp -Minfo=all -g #-acc -ta=tesla #-gpu=managed #-O3 -Mnovect
#-Wno-unused-result  


conv : conv_forward.o volume.o
	$(CC) $(CFLAGS) -o conv conv_forward.o  volume.o -lm


conv_forward.o : conv_forward.c volume.h
	$(CC) $(CFLAGS) -c conv_forward.c

volume.o : volume.c volume.h
	$(CC) $(CFLAGS) -c volume.c

clean:
	rm -f *.o
	rm -f conv_forward
	rm -f ./output/*.txt

.PHONY : clean
