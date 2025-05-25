/* --------------------------------------------------------------------------------

  �{���@�S��
�@���𐔂��邱�Ƃ��炢�ł��邳�F���������̊ȒP�Ȏ���

�@�� �R�}���h���C��

 �@�@Count <seed>
    �@�@�@ <seed> �F�����̎�     seed=13 �̏ꍇ���ł��悢(2007/02/21)

�@���̓��F����C�o�͂� 0 �� 1 �ƂȂ�̂ŁC�G���[�̎�������ł��C
�@����𗘗p���āC�o�͑w�̃��j�b�g�� 0.5 �ȏ�Ȃ� 1 �Ƃ��Ĉ����C
�@0.5 �����Ȃ� 0 �Ƃ��Ĉ����悤�ɂ����D

  ����͊m�F�ς݁D

  �@18416 �T�C�N���Ŏ����D

  Back Propagation �͐��K�̂܂܁D
  ---------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
//#include <process.h>
//#include <dos.h>

#define NoBits      6
#define NoOpes      1
#define NoIUnits    ( NoBits + NoOpes )
#define NoHUnits    7
#define NoOUnits    NoBits
#define NoPatterns  ( NoBits*2 )
#define NoTpat      14
#define Test        NoPatterns

#define Eta         0.75
#define Alpha       0.8
// Eta=0.6,Alpha=0.9,Seed=13
#define ErrorEv     0.45   //0.15
#define Rlow        -0.30
#define Rhigh       0.30

#define fout( x )   (1 / ( 1 + exp( -(x) ) ))
#define urand()     ( (double)rand() / 0x7fff * (Rhigh - Rlow) + Rlow )

#define MAXRDBUF    130

// �v���g�^�C�v�錾
void    forwardPropagation( int p );
void    backPropagation( int p );
void    initialize();
void    studyMode();
void    testMode();
int     operation();
void    printState( int cycle,int pat );
double  error( int pat );

// ���j�b�g�̐錾
double outIn[NoPatterns][NoIUnits];
double outHid[NoHUnits],outOut[NoOUnits];

// ���j�b�g�Ԃ̌����d��
double witoh[NoHUnits][NoIUnits],dwitoh[NoHUnits][NoIUnits];
double whtoo[NoOUnits][NoHUnits],dwhtoo[NoOUnits][NoHUnits];

// ���j�b�g�̃o�C�A�X
double hbias[NoHUnits],dhbias[NoHUnits];
double obias[NoOUnits],dobias[NoOUnits];

// �w�K�X�P�W���[�����O�e�[�u��
int  teachschedule[NoTpat] = { 6,8,4,13,2,11,0,3,12,1,10,9,5,7 };
// ���t�f�[�^
struct
  { int  tinput[NoIUnits];
    int  toutput[NoOUnits];
  } tsignal[NoTpat] =
  { { { 0,0,0,0,0,0, 1 },{ 1,0,0,0,0,0 } },
    { { 1,0,0,0,0,0, 1 },{ 0,1,0,0,0,0 } },
    { { 0,1,0,0,0,0, 1 },{ 0,0,1,0,0,0 } },
    { { 0,0,1,0,0,0, 1 },{ 0,0,0,1,0,0 } },
    { { 0,0,0,1,0,0, 1 },{ 0,0,0,0,1,0 } },
    { { 0,0,0,0,1,0, 1 },{ 0,0,0,0,0,1 } },
    { { 0,0,0,0,0,1, 1 },{ 0,0,0,0,0,0 } },

    { { 0,0,0,0,0,0, 0 },{ 0,0,0,0,0,1 } },
    { { 0,0,0,0,0,1, 0 },{ 0,0,0,0,1,0 } },
    { { 0,0,0,0,1,0, 0 },{ 0,0,0,1,0,0 } },
    { { 0,0,0,1,0,0, 0 },{ 0,0,1,0,0,0 } },
    { { 0,0,1,0,0,0, 0 },{ 0,1,0,0,0,0 } },
    { { 0,1,0,0,0,0, 0 },{ 1,0,0,0,0,0 } },
    { { 1,0,0,0,0,0, 0 },{ 0,0,0,0,0,0 } }
  };
/*
  { { { 0,0,0,0,0,0, 1,0 },{ 1,0,0,0,0,0 } },
    { { 1,0,0,0,0,0, 1,0 },{ 0,1,0,0,0,0 } },
    { { 0,1,0,0,0,0, 1,0 },{ 0,0,1,0,0,0 } },
    { { 0,0,1,0,0,0, 1,0 },{ 0,0,0,1,0,0 } },
    { { 0,0,0,1,0,0, 1,0 },{ 0,0,0,0,1,0 } },
    { { 0,0,0,0,1,0, 1,0 },{ 0,0,0,0,0,1 } },
    { { 0,0,0,0,0,1, 1,0 },{ 0,0,0,0,0,0 } },

    { { 0,0,0,0,0,0, 0,1 },{ 0,0,0,0,0,1 } },
    { { 0,0,0,0,0,1, 0,1 },{ 0,0,0,0,1,0 } },
    { { 0,0,0,0,1,0, 0,1 },{ 0,0,0,1,0,0 } },
    { { 0,0,0,1,0,0, 0,1 },{ 0,0,1,0,0,0 } },
    { { 0,0,1,0,0,0, 0,1 },{ 0,1,0,0,0,0 } },
    { { 0,1,0,0,0,0, 0,1 },{ 1,0,0,0,0,0 } },
    { { 1,0,0,0,0,0, 0,1 },{ 0,0,0,0,0,0 } }
  };
*/
int  ope[NoOpes];

int main(int argc, char* argv[])
{
  int seed = atoi( argv[1] );
  srand( seed );

  initialize();

  studyMode();
  testMode();

  return 0;
}

void forwardPropagation( int p )
{
  int  i,j;
  double sum;

  for( i = 0; i < NoHUnits; i++ )
    { sum = 0.0;
      for( j = 0; j < NoIUnits; j++ )
        sum += witoh[i][j]*outIn[p][j];
      outHid[i] = fout( sum + hbias[i] );
    }
  for( i = 0; i < NoOUnits; i++ )
    { sum = 0.0;
      for( j = 0; j < NoHUnits; j++ )
        sum += whtoo[i][j]*outHid[j];
      outOut[i] = fout( sum + obias[i] );
    }
}

void backPropagation( int p )
{
  int     i,j;
  double  dwih[NoHUnits],dwho[NoOUnits],sum,wk;

  for( i = 0; i < NoOUnits; i++ )
    { wk = outOut[i];
      dwho[i] = ( tsignal[p].toutput[i] - wk )*wk*( 1.0 - wk );
    }
  for( i = 0; i < NoHUnits; i++ )
    { for( sum = 0, j = 0; j < NoOUnits; j++ )
        { dwhtoo[j][i] = Eta*dwho[j]*outHid[i] + Alpha*dwhtoo[j][i];
          whtoo[j][i] += dwhtoo[j][i];
          sum += dwho[j]*whtoo[j][i];
        }
      dwih[i] = outHid[i]*( 1 - outHid[i] )*sum;
    }
  for( i = 0; i < NoOUnits; i++ )
    { dobias[i] = Eta*dwho[i] + Alpha*dobias[i];
      obias[i] += dobias[i];
    }

  for( i = 0; i < NoIUnits; i++ )
    { for( j = 0; j < NoHUnits; j++ )
        { dwitoh[j][i] = Eta*dwih[j]*outIn[p][i] + Alpha*dwitoh[j][i];
          witoh[j][i] += dwitoh[j][i];
        }
    }
  for( i = 0; i < NoHUnits; i++ )
    { dhbias[i] = Eta*dwih[i] + Alpha*dhbias[i];
      hbias[i] += dhbias[i];
    }
}
  
void initialize()
{
  int  i,j;

  for( i = 0; i < NoHUnits; i++ )
    for( j = 0; j < NoIUnits; j++ )
      witoh[i][j] = urand();

  for( i = 0; i < NoOUnits; i++ )
    for( j = 0; j < NoHUnits; j++ )
      whtoo[i][j] = urand();
}

void studyMode()
{
  int     cycle,scpat,pat,i;
  double  err;

  cycle = 0;
  while( true )
    { for( err = 0.0, scpat = 0; scpat < NoTpat; scpat++ )
        { pat = teachschedule[scpat];
          for( i = 0; i < NoIUnits; i++ )
            outIn[pat][i] = tsignal[pat].tinput[i];
          forwardPropagation( pat );
          printState( cycle,pat );
          backPropagation( pat );
          err += error( pat );
        }
      cycle++;
      printf( "  Error = %lf\n",err );
      if ( err <= ErrorEv ) break;
    } 
}

void testMode()
{
  int  i;

  for( i = 0; i < NoIUnits; i++ ) outIn[Test][i] = 0;

  while( operation() )
    { for( i = 0; i < NoOpes; i++ )
        outIn[Test][NoBits+i] = ope[i];

      forwardPropagation( Test );

      printf( "operator[ " );
      for( i = 0; i < NoOpes; i++ ) printf( "%d",ope[i] );
        printf( " ] -> " );
      for( i = 0; i < NoOUnits; i++ )
        { if ( outOut[i] >= 0.5 ) printf( "1 " );
          else                    printf( "0 " );
        }
      printf( "\n" );

      for( i = 0; i < NoBits; i++ )
        if ( outOut[i] >= 0.5 ) outIn[Test][i] = 1.0;
        else                    outIn[Test][i] = 0.0;
    }
}

int operation()
{
  int   onecemore,i;
  char  buf[MAXRDBUF];

  while( 1 )
    { buf[0] = NULL;
      printf( " ���얽��[%dBit] > ",NoOpes ); gets( buf );
      if ( strlen( buf ) == 0 ) return( 0 );

      onecemore = 0;
      for( i = 0; i < NoOpes; i++ )
        { if ( buf[i] != '0' && buf[i] != '1' ) onecemore = 1;
          ope[i] = buf[i] - '0';
        }
      if ( onecemore == 0 ) break;
    }

  return( 1 );
}

void printState( int cycle,int pat )
{
  int   i;

  printf( "\n Cycle:%4d-%d",cycle+1,pat+1 );
  for( i = 0; i < NoOUnits; i++ )
    printf( "\n Out-%d  %lf[ %d ]",i+1,outOut[i],tsignal[pat].toutput[i] );
}

double error( int pat )
{
  double  err,wk;
  int     i;

  err = 0.0;
  for( i = 0; i < NoOUnits; i++ )
    { // 2007/02/21 
      //wk = outOut[i] - tsignal[pat].toutput[i];
      if ( outOut[i] >= 0.5 ) wk = 1.0; else wk = 0.0;
      wk = wk - tsignal[pat].toutput[i];
      err += wk*wk;
    }

  err = err/NoOUnits;
  err = sqrt( err );

  return( err );
}

