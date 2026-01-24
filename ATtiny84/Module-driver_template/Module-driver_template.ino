#include <TinyWireS.h>
#define SLAVE_ADDR 0x20 // Example address

byte dataRecieved;

void setup(){
  TinyWireS.begin(SLAVE_ADDR);
  TinyWireS.onReceive(recieved);
  TinyWireS.onRequest(requested);
}

void loop(){
}

void requested(){
  TinyWireS.write(0x32); // Send data to master
}

void recieved(uint8_t data){
}