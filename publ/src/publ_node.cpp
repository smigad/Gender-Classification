#include "portaudio.h"
#include <stdio.h>
#include <stdlib.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/MultiArrayLayout.h"
#include "std_msgs/MultiArrayDimension.h"
#include "std_msgs/Int16MultiArray.h"
#include "std_msgs/Int16.h"
#include "publ/aud_publ.h"
#include <stdint.h>


#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 512
#define NUM_SECONDS 10
#define NUM_CHANNELS 1

#define WRITE_TO_FILE 0

#define ENABLE_PLAYBACK 0

//--------------- SETTING SAMPLE FORMAT
#define PA_SAMPLE_TYPE paInt16
typedef short SAMPLE; //------each sample is a signed short integer 


ros::Publisher pub;


PaError err; //return code for most PA functions... if not PaNoError then there is error
PaStream *o_stream;
int cntr = 0;
//struct for holding data
typedef struct {
  int frame_idx;
  int max_frame_idx;
  SAMPLE *recorded_samples;
} RecData;


//recorder input callback
static int recorder(const void *i_buff, void *o_buff, unsigned long frames_per_buffer, 
          const PaStreamCallbackTimeInfo *time_info, PaStreamFlags status_flag, 
          void *i_data)
{
  RecData *data = (RecData*) i_data;
  const SAMPLE *rptr = (const SAMPLE*) i_buff;
  SAMPLE *wptr = &data->recorded_samples[data->frame_idx * NUM_CHANNELS];
  long frames_to_calc;
  int rtrn;
  unsigned long frames_left = data->max_frame_idx - data->frame_idx;
  (void) o_buff; (void) time_info; (void)status_flag; (void) i_data;
  


  if(frames_left < frames_per_buffer){
    frames_to_calc = frames_left;
    rtrn = paComplete;
  }
  else{
    frames_to_calc = frames_per_buffer;
    rtrn = paContinue;
  }


  publ::aud_publ audio_data;
  std_msgs::Int16 i16_audio_data;

  publ::aud_publ the_array;


  int16_t *dat = (int16_t *) i_buff;
  for (int i = 0; i < frames_to_calc; i++){
    printf("%d ", dat[i]);
    i16_audio_data.data = dat[i];
    //audio_data.audio_raw.push_back(i16_audio_data);
    the_array.audio_raw_data.data.push_back(dat[i]);
  }
  printf("---------------------------------------------\n");
    printf("FRAMES: %ld\nSIZE: %lu\n", frames_to_calc, sizeof(i_buff)); 
    if(ENABLE_PLAYBACK)
      err = Pa_WriteStream(o_stream, i_buff, frames_to_calc);
  data->frame_idx += frames_to_calc;
  pub.publish(the_array);
  the_array.audio_raw_data.data.clear();
  return rtrn;
}



int main(int argc, char **argv)
{
  printf("formats==\npaInt16 = %lu\npaInt24 = %lu\npaInt32 = %lu\npaFloat32 = %lu\n",
    sizeof(paInt16), sizeof(int64_t), sizeof(paInt32), sizeof(paFloat32));
  //init ros
  ros::init(argc, argv, "audio_publisher");
  ros::NodeHandle n;
  //pub = n.advertise<std_msgs::Int16MultiArray>("audio_pub", 10);
  pub = n.advertise<publ::aud_publ>("audio_pub", 10);

  int num_device, def_i_device, def_o_device;
  const PaDeviceInfo *device_info; //structure describing PA device
  PaStreamParameters i_params, o_params; //io parameters for stream... if one then the other must be null
  PaStream *stream;
  RecData data, udata;
  int total_frames, num_samples, num_bytes;

  //total number of frames
  total_frames = NUM_SECONDS * SAMPLE_RATE;
  num_samples = total_frames * NUM_CHANNELS;
  num_bytes = num_samples * sizeof(SAMPLE);

  printf("TOTAL FRAMES: %d\nNUM. SAMPLES: %d\nNUM. BYTES: %d\n", total_frames, num_samples, num_bytes);

  data.max_frame_idx = total_frames;
  data.frame_idx = 0;
  data.recorded_samples = (SAMPLE*) malloc(num_bytes); //check if memory allocated
  if (data.recorded_samples == NULL){
    printf("ERROR!!!! COULD NOT ALLOCATE MEMORY\n");
    err = -1;
    goto error_point;
  } else printf("MEMORY CORRECTLY ALLOCATED\n");

  //SET recorded_samples ARRAY TO 0
  for (int i = 0; i < num_samples; i++) data.recorded_samples[i] = 0;

  //initializing pa
  err = Pa_Initialize();
  if(err != paNoError){//if error occured
    printf("ERROR!! Initialization failed!\n");
    goto error_point;
  }

  //if no error
  printf("portaudio version: 0x%08X\n", Pa_GetVersion());
  printf("Version text: %s\n", Pa_GetVersionText());

  num_device = Pa_GetDeviceCount();

  if(num_device < 0){
    printf("ERROR Pa_GetDeviceCount() returned %d\n", num_device);
      err = num_device;
      goto error_point;
  }

  //if num_device is fine
  printf("Number of Devices: %d\n", num_device);
  def_i_device = Pa_GetDefaultInputDevice();
  def_o_device = Pa_GetDefaultOutputDevice();
  printf("Default Input Device ID: %d\n", def_i_device);
  printf("Default Output Device ID: %d\n", def_o_device);

  //getting info on each of the devices
  for (int i = 0; i < num_device; i++){
    device_info = Pa_GetDeviceInfo(i);
    
    if (i == Pa_GetDefaultInputDevice())
      printf("---- DEFAULT INPUT DEVICE!! ----\n");
    if (i == Pa_GetDefaultOutputDevice())
      printf("**** DEFAULT OUTPUT DEVICE!! ****\n");
    


    printf("Device Name: \t%s\n", device_info->name);
    printf("Host API: \t%s\n", Pa_GetHostApiInfo(device_info->hostApi)->name);
    printf("Max Inputs: \t%d\n", device_info->maxInputChannels);
    printf("Max Outputs: \t%d\n", device_info->maxOutputChannels);
    printf("Default low input latency   = %8.4f\n", device_info->defaultLowInputLatency  );
        printf("Default low output latency  = %8.4f\n", device_info->defaultLowOutputLatency  );
        printf("Default high input latency  = %8.4f\n", device_info->defaultHighInputLatency  );
        printf("Default high output latency = %8.4f\n", device_info->defaultHighOutputLatency  );
        printf("Default Sample Rate: \t%8.2f\n", device_info->defaultSampleRate);
  }//done displaying 

  //USE FUNCTION Pa_IsFormatSupported(iparams, oparams, sample_rate)
  //RETURNS paFormatIsSupported IF THE SAMPLE RATE IS SUPPORTED
  //use the default device
  device_info = Pa_GetDeviceInfo(def_i_device);

  i_params.device = def_i_device;
  i_params.channelCount = NUM_CHANNELS;
  i_params.sampleFormat = paInt16;
  i_params.suggestedLatency = 0; //hehe I wish
  i_params.hostApiSpecificStreamInfo = NULL;
  
  if(ENABLE_PLAYBACK){
    o_params.device = def_o_device;
    o_params.channelCount = NUM_CHANNELS;
    o_params.sampleFormat = paInt16;
    o_params.suggestedLatency = 0;
    o_params.hostApiSpecificStreamInfo = NULL;
  }

  //OPENING INPUT STREAM
  err = Pa_OpenStream(&stream, &i_params, NULL, SAMPLE_RATE, 
    FRAMES_PER_BUFFER, paClipOff, recorder, &data);

  if(err != paNoError){
    printf("ERROR!! COULD NOT OPEN INPUT STREAM -- %d\n", err);
    goto error_point;
  }

  if(ENABLE_PLAYBACK){
      //OPENING OUTPUT STREAM
      err = Pa_OpenStream(&o_stream, NULL, &o_params, SAMPLE_RATE, 
        FRAMES_PER_BUFFER, paClipOff, NULL, &udata);
      if(err != paNoError){
        printf("ERROR!! COULD NOT OPEN OUTPUT STREAM -- %d\n", err);
        goto error_point;
      }
  }



  err = Pa_StartStream(stream);
  if(err != paNoError)
  {
    printf("ERROR!! COULD NOT START INPUT STREAM -- %d\n", err);
    goto error_point;
  }
  else printf("*** RECORDING ***\n");


  if(ENABLE_PLAYBACK)
  {
      //START OUTPUT STREAM
      err = Pa_StartStream(o_stream);
      if(err != paNoError)
      {
        printf("ERROR!! COULD NOT START OUTPUT STREAM -- %d\n", err);
        goto error_point;
      }
      else printf("*** PLAYBACK ***\n");
  }


  while(Pa_IsStreamActive(stream)) //while the stream is active
  {
    //maybe do some delays or whatever and output some shit
    Pa_Sleep(1000);
    printf("INDEX: %d\n", data.frame_idx);
  }

  err = Pa_CloseStream(stream);
  if(err != paNoError)
  {
    printf("ERROR!! COULD NOT CLOSE INPUT STREAM -- %d\n", err);
    goto error_point;
  }
  
  if(ENABLE_PLAYBACK)
  {
      err = Pa_CloseStream(o_stream);
      if(err != paNoError)
      {
        printf("ERROR!! COULD NOT CLOSE OUTPUT STREAM -- %d\n", err);
        goto error_point;
      }
  }


  #if WRITE_TO_FILE 
  {
    FILE *out_file;
    out_file = fopen("recorded_raw.raw", "wb");
    if(!out_file){
      printf("ERROR!! COULD NOT OPEN FILE FOR WRITING\n");
      goto error_point;
    }
    
    fwrite(data.recorded_samples, NUM_CHANNELS*sizeof(SAMPLE), total_frames, out_file);
    fclose(out_file);
    printf("WRITING TO FILE SUCCESSFUL!!\n");
  }
  #endif


  Pa_Terminate();

//FREE UP MEMORY
  if(data.recorded_samples != NULL) free(data.recorded_samples);

  printf("EXITING.... MEMORY FREED\n");
  return 0;


error_point:
  Pa_Terminate();
  printf("Exiting... ERROR CODE: %s\n",  Pa_GetErrorText(err));
  return err;
}


