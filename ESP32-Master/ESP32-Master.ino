#include <Wire.h>

#define slaveAddress 0x20
#define SDA 21
#define SCL 22

byte data_received;

void setup()
{
  Serial.begin(9600);
  Wire.begin(SDA, SCL);

  Wire.beginTransmission(slaveAddress);
  byte busStatus = Wire.endTransmission();
  if (busStatus != 0)
  {
    Serial.println("Slave is not found.");
    while (true);
  }
  Serial.println("Slave is found.");
}

void loop()
{
  // 1.
  Wire.beginTransmission(slaveAddress); // Start communication with slave address
  Wire.write(0x01);                     // Send the internal register address to read from
  Wire.endTransmission(false);          // Send a repeated start condition after this write (parameter false)

  // 2. Request data from the slave (a read transaction)
  uint8_t bytesRequested = 1; // Number of bytes to request
  Wire.requestFrom(slaveAddress, bytesRequested); // Request bytes from the slave

  // 3. Read the data
  while (Wire.available()) { // While data is available to read
    data_received = Wire.read(); // Read a byte
    Serial.print("Received data: ");
    Serial.println(data_received);
  }

  delay(500);
}