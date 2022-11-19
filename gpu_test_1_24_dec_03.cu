#include <iostream>
#include <ctime>
#include <stdio.h>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <bits/stdc++.h>

#include <string>
#include <cmath>

using namespace std ;
fstream file_9 ;
fstream file_4 ;
fstream file_3 ;
fstream file_18 ;
fstream file_13;

void ALLOCATE_GAS();
void HARD_SPHERE();
void ARGON();
void IDEAL_NITROGEN();
void REAL_OXYGEN();
void IDEAL_AIR();
void REMOVE_MOL(int &) ;
void SAMPLE_FLOW() ;
void OUTPUT_RESULTS() ;
void SETXT() ;
void cuda_collisions(int) ;
void COLLISIONS() ;
void DISSOCIATION() ;
void EXTEND_MNM(double ) ;
void MOLECULES_ENTER_1D() ;
void INDEX_MOLS() ;
void AIFX(double &,double &, double &, double &, double &, double &, double &, double &) ;
void RBC(double &, double &, double &,double &, double &,double &) ;
void REAL_AIR();
void HELIUM_ARGON_XENON();
void OXYGEN_HYDROGEN();
void MOLECULES_MOVE_1D() ;
void DERIVED_GAS_DATA() ;
void INITIALISE_SAMPLES();
void SET_INITIAL_STATE_1D();
void FIND_CELL_MB_1D(double& , int& ,int& , double&) ;
void FIND_CELL_1D(double &,int &,int &);
void RVELC(double &,double &,double &);
void SROT(int &,double &,double &);
void SVIB(int &,double &,int &, int&);
void SELE(int &,double &,double &);
void CQAX(double&,double &,double&);
void LBS(double,double,double&);
void ENERGY(int ,double &);
void REFLECT_1D(int&,int,double&);
void READ_DATA();


//modules calc
int NVER ,MVER,IMEG,NREL,MOLSC,ISF,ISAD,ISECS,IGS,IREM,NNC,IMTS,ERROR,NLINE,ICLASS, 
           NCLASS,NMCC,NMI,NMP,ICN ;
double FTIME,TLIM,PI,SPI,DPI,BOLTZ,FNUM,DTM,TREF,TSAMP,TOUT,SAMPRAT,OUTRAT,RANF,TOTCOLI,TOTMOVI,TENERGY,
                 DTSAMP,DTOUT,TPOUT,FRACSAM,TOTMOV,TOTCOL,ENTMASS,ENTREM,CPDTM,TPDTM,AVOG,TNORM,FNUMF;
double *VNMAX , *TDISS , *TRECOMB , *ALOSS , *EME , *AJM , *COLL_TOTCOL;
double *TCOL ; 

//module molecs
int *IPCELL,*IPSP,*ICREF,*IPCP ;
int *IPVIB ;
int NM , MNM ;
double *PX , *PV ;
double *PTIM , *PROT , *PELE  ;

//module gas
double RMAS,CXSS,RGFS,VMPM,FDEN,FPR,FMA,FPM,CTM ;
double FND[3],FTMP[3],FVTMP[3],VFX[3],VFY[3],TSURF[3],FSPEC[3],VSURF[3] ;
double *ERS,*CR,*TNEX,*PSF,*SLER,*FP ;
double *FSP,*SP,*SPR,*SPV,*VMP ;
double *SPM,*SPVM,*ENTR,*QELC,*SPRT ;
double *SPEX,*SPRC,*SPRP ;
double *SPREX ;

int MSP,MMVM,MMRM,MNSR,IGAS,MMEX,MEX,MELE,MVIBL ;
int *ISP,*ISPV,*NELL ;
int *ISPR,*LIS,*LRS,*ISRCD,*ISPRC,*ISPRK,*TREACG,*TREACL,*NSPEX,*NSLEV ;
int *ISPVM,*NEX ;
int *ISPEX ;

//module OUTPUT
int NSAMP,NMISAMP,NOUT,NDISSOC,NRECOMB,NTSAMP ;
int *NDISSL ;
double TISAMP , XVELS , YVELS , AVDTM ;
double *COLLS,*WCOLLS,*CLSEP,*SREAC,*STEMP,*TRANSTEMP,*ROTTEMP,*VIBTEMP,*ELTEMP ;
double *VAR,*VARS,*CSSS,*SUMVIB ;
double *CS,*VARSP,*VIBFRAC ;
double *CSS ;


//module GEOM_1D
int NCELLS,NCCELLS,NCIS,NDIV,MDIV,ILEVEL,IFX,JFX,IVB,IWF ;
int *ITYPE ;
int *ICELL ;
int *ICCELL , *JDIV ;
double DDIV,XS,VELOB,WFM,AWF,FREM,XREM ;
double *XB ;
double *CELL , *CCELL ;


void i_allocate( int x , int y , int *&b)
{
        
        
        cudaMallocManaged(&b , (x*y+6)*sizeof(int) ) ;
        //b = new int[x*y+6] ;
        cudaMemset(b, 0 , (x*y+6)*sizeof(int)) ;
        b[0] = x*y ;
        b[b[0]+1] = x ;
        b[b[0]+2] = y ;
}

void i_allocate( int x , int *&b )
{
      
        cudaMallocManaged(&b , (x+6)*sizeof(int) ) ;
        //b = new int[x+6] ;
        cudaMemset(b, 0 , (x+1)*sizeof(int)) ;
        b[0] = x ;
        b[b[0]+1]=x ;
}

void i_allocate( int x , int y , int z , int *&b )
{
        
        cudaMallocManaged(&b , (x*y*z+6)*sizeof(int) ) ;    
        //b= new int[x*y*z+6] ;
        cudaMemset(b, 0 , (x*y*z+6)*sizeof(int)) ;
        b[0] = x*y*z ;
        b[b[0]+1] = x ;
        b[b[0]+2] = y ;
        b[b[0]+3] = z ;
}

void i_allocate( int x , int y , int z , int w , int *&b  )
{
        
        cudaMallocManaged(&b , (x*y*z*w+6)*sizeof(int) ) ;
        //b = new int[x*y*z*w+6] ;
        cudaMemset(b, 0 , (x*y*z*w+6)*sizeof(int)) ;
        b[0] = x*y*z*w ;
        b[b[0]+1] = x ;
        b[b[0]+2] = y ;
        b[b[0]+3] = z ;
        b[b[0]+4] = w ;
}

void i_allocate( int x , int y , int z , int w ,int u , int *&b )
{
        
        cudaMallocManaged(&b , (x*y*z*w*u+6)*sizeof(int) ) ;
        //b = new int[x*y*z*w*u+6] ;
        cudaMemset(b, 0 , (x*y*z*w*u+6)*sizeof(int)) ;
        b[0] = x*y*z*w*u ;
        b[b[0]+1] = x ;
        b[b[0]+2] = y ;
        b[b[0]+3] = z ;
        b[b[0]+4] = w ;
        b[b[0]+5] = u ;
}

//__device__ __host__ 


//__device__ __host__

int& get(int* &b , int x)
{
    return b[x] ;
} 
int& get(int *&b , int x , int y)
{
        int r1 = b[b[0]+2] ;
        return b[(x-1)*r1 + y] ;
}

//__device__ __host__ 
int& get(int *&b , int x , int y , int z)
{
        int r1 = b[b[0]+2] ;
        int r2 = b[b[0]+3] ;
        return b[((x-1)*r1 + y-1)*r2+z] ;
}

//__device__ __host__ 
int& get(int *&b , int x , int y , int z , int w)
{
        int r1 = b[b[0]+2] ;
        int r2 = b[b[0]+3] ;
        int r3 = b[b[0]+4] ;
        return b[(((x-1)*r1 + y-1)*r2+z-1)*r3+w] ;
}

//__device__ __host__ 
int& get(int *&b , int x , int y , int z , int w , int u)
{
        int r1 = b[b[0]+2] ;
        int r2 = b[b[0]+3] ;
        int r3 = b[b[0]+4] ;
        int r4 = b[b[0]+5] ;
        return b[((((x-1)*r1 + y-1)*r2+z-1)*r3+w-1)*r4+u] ;
}


void d_allocate( int x , int y , double *&b )
{
        
        cudaMallocManaged(&b , (x*y+6)*sizeof(double) ) ;
        //b = new double[x*y+6] ;
        cudaMemset(b, 0 , (x*y+6)*sizeof(double)) ;
        b[0] = (double)x*y ;
        b[(int)b[0]+1] = x ;
        b[(int)b[0]+2] = y ;
}

void d_allocate( int x  , double *&b )
{
       
        cudaMallocManaged(&b , (x+1)*sizeof(double) ) ;
        //b = new double[x+6] ;
        cudaMemset(b, 0 , (x+1)*sizeof(double)) ;
        b[0] = x ;
        b[(int)b[0]+1] = x ;
}

void d_allocate( int x , int y , int z , double *&b )
{
       
        cudaMallocManaged(&b , (x*y*z+6)*sizeof(double) ) ;
        //b = new double[x*y*z+6] ;
        cudaMemset(b, 0 , (x*y*z+6)*sizeof(double)) ;
        b[0] = x*y*z ;
        b[(int)b[0]+1] = x ;
        b[(int)b[0]+2] = y ;
        b[(int)b[0]+3] = z ;
}

void d_allocate( int x , int y , int z , int w ,double *&b )
{
      
        cudaMallocManaged(&b , (x*y*z*w+6)*sizeof(double) ) ;
        //b = new double[x*y*z*w+6] ;
        cudaMemset(b, 0 , (x*y*z*w+6)*sizeof(double)) ;
        b[0] = x*y*z*w ;
        b[(int)b[0]+1] = x ;
        b[(int)b[0]+2] = y ;
        b[(int)b[0]+3] = z ;
        b[(int)b[0]+4] = w ;
}

void d_allocate(int x , int y , int z , int w ,int u , double *&b )
{
        cudaMallocManaged(&b , (x*y*z*w*u+6)*sizeof(double) ) ;
        //b = new double[x*y*z*w*u+6] ;
        cudaMemset(b, 0 , (x*y*z*w*u+6)*sizeof(double)) ;
        b[0] = x*y*z*w*u ;
        b[(int)b[0]+1] = x ;
        b[(int)b[0]+2] = y ;
        b[(int)b[0]+3] = z ;
        b[(int)b[0]+4] = w ;
        b[(int)b[0]+5] = u ;
}

//__device__ __host__ 
double& get(double *&b , int x)
{
        return b[x] ;
}

//__device__ __host__ 
double& get(double *&b , int x , int y)
{
        int r1 = b[(int)b[0]+2] ;
        return b[(x-1)*r1 + y] ;
}

//__device__ __host__ 
double& get(double *&b , int x , int y , int z)
{
        int r1 = b[(int)b[0]+2] ;
        int r2 = b[(int)b[0]+3] ;
        return b[((x-1)*r1 + y-1)*r2+z] ;
}

//__device__ __host__ 
double& get(double *&b , int x , int y , int z , int w)
{
        int r1 = b[(int)b[0]+2] ;
        int r2 = b[(int)b[0]+3] ;
        int r3 = b[(int)b[0]+4] ;
        return b[(((x-1)*r1 + y-1)*r2+z-1)*r3+w] ;
}

//__device__ __host__ 
double& get(double *&b , int x , int y , int z , int w , int u)
{
        int r1 = b[(int)b[0]+2] ;
        int r2 = b[(int)b[0]+3] ;
        int r3 = b[(int)b[0]+4] ;
        int r4 = b[(int)b[0]+5] ;
        return b[((((x-1)*r1 + y-1)*r2+z-1)*r3+w-1)*r4+u] ;
}

         


void READ_DATA()
{
    int NVERD , MVERD , N , K ;
    d_allocate(2 , XB ) ;
    i_allocate(2 , ITYPE ) ;
    i_allocate(201 , NDISSL ) ;
    cout << "ICLASS = "<< ICLASS << endl ;
    if(ICLASS==0)
        {
            cout << "Reading the data file DS0D.DAT\n";
            file_4.open("DS0D.DAT" , ios::in) ;
            file_3.open("DS0D.TXT" , ios::out) ;
            file_3 << "Data summary for program DSMC" << endl ;
            cout  << "DS0D.TXT opened \n" ;
        }

    if(ICLASS==1)
    {
        cout<<"Reading the data file DS1D.DAT"<<endl;
        file_4.open("DS1D.DAT", ios::in);
        file_3.open("DS1D.TXT", ios::out );
        file_3<<"Data summary for program DSMC"<<endl;
        // OPEN (4,FILE='DS1D.DAT')
        // OPEN (3,FILE='DS1D.TXT')
        // WRITE (3,*) 'Data summary for program DSMC'
    }

    file_4 >> IMEG ;
    file_3 << "The approximate number of megabytes for the calculation is" << IMEG  << endl ;
    file_4 >> IGAS ;
    file_3<< IGAS << endl;

    if(IGAS==1)
    {
        file_3<<" Hard sphere gas "<<endl;
        // WRITE (3,*) 'Hard sphere gas'
        HARD_SPHERE();
    }
    else if(IGAS==2)
    {
        file_3<<"Argon "<<endl;
        // WRITE (3,*) 'Argon'
        ARGON();
    }
    else if(IGAS==3)
    {
        file_3<<"Ideal nitrogen"<<endl;
        // WRITE (3,*) 'Ideal nitrogen'
        IDEAL_NITROGEN();
        
    }
    else if(IGAS==4)
    {
        file_3<<"Real oxygen "<<endl;
        // WRITE (3,*) 'Real oxygen'
        REAL_OXYGEN();
    }
    else if(IGAS==5)
    {
        file_3<<"Ideal air "<<endl;
        // TE (3,*) 'Ideal air'
        IDEAL_AIR();
    }
    else if(IGAS==6)
    {
        file_3<<"Real air @ 7.5 km/s "<<endl;
        // RITE (3,*) 'Real air @ 7.5 km/s'
        REAL_AIR();
    }
    else if(IGAS==7)
    {
        file_3<<"Helium-argon-xenon mixture "<<endl;
        // WRITE (3,*) 'Helium-argon-xenon mixture'
        HELIUM_ARGON_XENON();
    }
    else if(IGAS==8)
    {
        file_3<<"Oxygen-hydrogen "<<endl;
        // WRRITE (3,*) 'Oxygen-hydrogen'
        OXYGEN_HYDROGEN();
    }

  
    

    
    file_3<<"The gas properties are:- "<<endl;
    file_4>>FND[1];
    file_3<<"The stream number density is "<<FND[1]<<endl;
    file_4>>FTMP[1];
    file_3<<"The stream temperature is "<<FTMP[1]<<endl;

    

    if(MMVM>0)
    {
      
        file_4>>FVTMP[1];
        file_3<<"The stream vibrational and electronic temperature is "<<FVTMP[1]<<endl;
        // READ (4,*) FVTMP(1) //FVTMP;
        // WRITE (3,*) '    The stream vibrational and electronic temperature is',FVTMP(1) //FVTMP[1]
    }

    if(ICLASS==1)
    {
        file_4>>VFX[1];
        file_3<<"The stream velocity in the x direction is "<<VFX[1]<<endl;
        file_4>>VFY[1];
        file_3<<"The stream velocity in the y direction is "<<VFY[1]<<endl;
        // WRITE (3,*) '    The stream velocity in the y direction is',VFY(1) ////VFY[1]
    }

    if(MSP>1)
    {
        for(N=1;N<=MSP;N++)
        {   
            int in ;
            file_4 >> in;
            get(FSP ,N,1) =  in ;
            file_3 << " The fraction of species " << N <<" is "<<get(FSP ,N,1)<<endl;
            // WRITE (3,*) '    The fraction of species',N,' is',FSP(N,1) //get(FSP ,N,1]
        }
    }
    else get(FSP ,1,1) = 1 ;

    if(ICLASS==0){
        //       !--a homogeneous gas case is calculated as a one-dimensional flow with a single sampling cell
        // !--set the items that are required in the DS1D.DAT specification
        IFX=0;
        JFX=1;
        XB[1]=0.e00;
        XB[2]=0.0001e00*1.e25/FND[1];
        ITYPE[1]=1;
        ITYPE[2]=1;
        VFX[1]=0.e00;
        IGS=1;
        ISECS=0;
        IREM=0;
        MOLSC=10000*IMEG; //a single sampling cell
    }
    else if(ICLASS==1)
    {
        file_4>>IFX;
        
        if(IFX==0)
            file_3<<"Plane Flow"<<endl;
        
        if(IFX==1)
            file_3<<"Cylindrical flow"<<endl;
        
        if(IFX==2)
            file_3<<"Spherical flow"<<endl;
        
        JFX=IFX+1;
        file_4>>XB[1];
        
        file_3<<"The minimum x coordinate is "<<XB[1]<<endl;
        file_4>>ITYPE[1];
        if(ITYPE[1]==0)
            file_3<<"The minimum x coordinate is a stream boundary"<<endl;
        if(ITYPE[1]==1)
            file_3<<"The minimum x coordinate is a plane of symmetry"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is a plane of symmetry'
        if(ITYPE[1]==2)
            file_3<<"The minimum x coordinate is a solid surface"<<endl;
        if(ITYPE[1]==3)
            file_3<<"The minimum x coordinate is a vacuum"<<endl;
        if(ITYPE[1]==4)
            file_3<<"The minimum x coordinate is an axis or center"<<endl;
        if(ITYPE[1]==2)
        {
            file_3<<"The minimum x boundary is a surface with the following properties"<<endl;
            file_4>>TSURF[1];
            file_3<<"The temperature of the surface is "<<TSURF[1]<<endl;
            file_4>>FSPEC[1];
            file_3<<"The fraction of specular reflection is "<<FSPEC[1]<<endl;
            file_4>>VSURF[1];
            file_3<<"The velocity in the y direction of this surface is "<<VSURF[1] << endl ;
        }
        file_4>>XB[2];
        file_3<<"The maximum x coordinate is "<<XB[2]<<endl;
        file_4>>ITYPE[2];
        if(ITYPE[2]==0)
            file_3<<"The mmaximum  x coordinate is a stream boundary"<<endl;
        if(ITYPE[2]==1)
            file_3<<"The maximum x coordinate is a plane of symmetry"<<endl;
        if(ITYPE[2]==2)
            file_3<<"The maximum  x coordinate is a solid surface"<<endl;
        if(ITYPE[2]==3)
            file_3<<"The maximum  x coordinate is a vacuum"<<endl;
        ICN=0;
        if(ITYPE[2]==4)
        {
            file_3<<"The maximum x coordinate is a stream boundary with a fixed number of simulated molecules"<<endl;
            // WRITE (3,*) 'The maximum x coordinate is a stream boundary with a fixed number of simulated molecules'
            if(MSP==1)
                ICN=1;
        }
        if(ITYPE[2]==2)
        {
            file_3<<"The maximum  x boundary is a surface with the following properties"<<endl;
            file_4>>TSURF[1];
            file_3<<"The temperature of the surface is "<<TSURF[1]<<endl;
            file_4>>FSPEC[1];
            file_3<<"The fraction of specular reflection is "<<FSPEC[1]<<endl;
            file_4>>VSURF[1];
            file_3<<"The velocity in the y direction of this surface is "<<VSURF[1]<<endl;
        }
        if(IFX>0)
        {
            file_4>>IWF;
            if(IWF==0)
                file_3<<"There are no radial weighting factors"<<endl;
            if(IWF==1)
                file_3<<"There are radial weighting factors"<<endl;
            if(IWF==1)
            {
                file_4>>WFM;
                file_3<<"The maximum value of the weighting factor is  "<<WFM<<endl;
                WFM=(WFM-1)/XB[2];
            }
        }
        file_4>>IGS;
        // READ (4,*) IGS //IGS
        if(IGS==0)
            file_3<<"The flowfield is initially a vacuum "<<endl;
        // WRITE (3,*) 'The flowfield is initially a vacuum'
        if(IGS==1)
            file_3<<"The flowfield is initially the stream(s) or reference gas"<<endl;
        // WRITE (3,*) 'The flowfield is initially the stream(s) or reference gas'
        file_4>>ISECS;
        // READ (4,*) ISECS //ISECS
        if(ISECS==0)
            file_3<<"There is no secondary stream initially at x > 0"<<endl;
        // WRITE (3,*) 'There is no secondary stream initially at x > 0'
        if(ISECS==1 && IFX==0)
            file_3<<"There is a secondary stream applied initially at x = 0 (XB(2) must be > 0)"<<endl;
        // WRITE (3,*) 'There is a secondary stream applied initially at x = 0 (XB(2) must be > 0)'
        if(ISECS==1 && IFX>0)
        {
            if(IWF==1)
            {
                file_3<<"There cannot be a secondary stream when weighting factors are present"<<endl;
                // WRITE (3,*) 'There cannot be a secondary stream when weighting factors are present'
                return;//STOP//dout
            }
            file_3<<"There is a secondary stream"<<endl;
            // WRITE (3,*) 'There is a secondary stream'
            file_4>>XS;
            // READ (4,*) XS //XS
            file_3<<"The secondary stream boundary is at r= "<<XS<<endl;
            // WRITE (3,*) 'The secondary stream boundary is at r=',XS //XS
        }
        if(ISECS==1)
        {
            file_3<<"The secondary stream (at x>0 or X>XS) properties are:-"<<endl;
            file_4>>FND[2];
            file_3<<"The stream number density is "<<FND[2]<<endl;
            file_4>>FTMP[2];
            file_3<<"The stream temperature is "<<FTMP[2]<<endl;
            // WRITE (3,*) 'The secondary stream (at x>0 or X>XS) properties are:-'
            // READ (4,*) FND(2) //FND
            // WRITE (3,*) '    The stream number density is',FND(2) //FND
            // READ (4,*) FTMP(2) //FTMP
            // WRITE (3,*) '    The stream temperature is',FTMP(2) //FTMP
            if(MMVM>0)
            {
                file_4>>FVTMP[2];
                file_3<<"The stream vibrational and electronic temperature is "<<FVTMP[2]<<endl;
                // READ (4,*) FVTMP(2) //FVTMP[2]
                // WRITE (3,*) '    The stream vibrational and electronic temperature is',FVTMP(2) //FVTMP[2]
            }
            file_4>>VFX[2];
            file_3<<"The stream velocity in the x direction is "<<VFX[2]<<endl;
            file_4>>VFY[2];
            file_3<<"The stream velocity in the y direction is "<<VFY[2]<<endl;
            // READ (4,*) VFX(2) //VFX
            // WRITE (3,*) '    The stream velocity in the x direction is',VFX(2) //VFX
            // READ (4,*) VFY(2) //VFY
            // WRITE (3,*) '    The stream velocity in the y direction is',VFY(2) //VFY
            if(MSP>1)
            {
                for(N=1;N<=MSP;N++)
                {
                    int in ;
                    file_4>>in;
                    get(FSP ,N,2 )= in ;
                    file_3<<"The fraction of species "<<N<<" is "<<get(FSP ,N,2)<<endl;
                    // READ (4,*) FSP(N,2) //FSP
                    // WRITE (3,*) '    The fraction of species',N,' is',FSP(N,2) //FSP
                }
            }
            else
            {
                get(FSP ,1,2)=1;
            }
        }
        if(IFX==0 && ITYPE[1]==0)
        {
            file_4>>IREM;
            // READ (4,*) IREM //IREM
            if(IREM==0)
            {
                file_3<<"There is no molecule removal"<<endl;
                // WRITE (3,*) 'There is no molecule removal'
                XREM=XB[1]-1.e00;
                FREM=0.e00;
            }
            else if(IREM==1)
            {
                file_4>>XREM;
                file_3<<"There is full removal of the entering (at XB(1)) molecules between "<<XREM<<" and "<<XB[2]<<endl;
                // READ (4,*) XREM //XREM
                // WRITE (3,*) ' There is full removal of the entering (at XB(1)) molecules between',XREM,' and',XB(2) //XREM ,XB[2]
                FREM=1.e00;
            }
            else if(IREM==2)
            {
                file_3<<"Molecule removal is specified whenever the program is restarted"<<endl;
                // WRITE (3,*) ' Molecule removal is specified whenever the program is restarted'
                XREM=XB[1]-1.e00;
                FREM=0.e00;
            }
            else
            {
                XREM=XB[1]-1.e00;
                FREM=0.e00;
            }
        }
        //IVB=0;
        //VELOB=0.e00;
        if(ITYPE[2]==1)
        {
            file_4>>IVB;
            // READ (4,*) IVB
            if(IVB==0)
                file_3<<"The outer boundary is stationary"<<endl;
            // WRITE (3,*) ' The outer boundary is stationary'
            if(IVB==1)
            {
                file_3<<"The outer boundary moves with a constant speed"<<endl;
                file_4>>VELOB;
                file_3<<" The speed of the outer boundary is "<<VELOB<<endl;
                // WRITE (3,*) ' The outer boundary moves with a constant speed'
                // READ (4,*) VELOB //VELOB
                // WRITE (3,*) ' The speed of the outer boundary is',VELOB //VELOB
            }
        }
        file_4>>MOLSC;
        file_3<<"The desired number of molecules in a sampling cell is "<<MOLSC<<endl;
        // READ (4,*) MOLSC //MOLSC
        // WRITE (3,*) 'The desired number of molecules in a sampling cell is',MOLSC ////MOLSC
    }
    //set the speed of the outer boundary
    file_3.close();
    file_4.close();
    // CLOSE (3)
    // CLOSE (4)
    // set the stream at the maximum x boundary if there is no secondary stream
    if(ISECS==0 && ITYPE[2]==0)
    {
        FND[2]=FND[1];
        FTMP[2]=FTMP[1];
        if(MMVM>0)
            FVTMP[2]=FVTMP[1];
        VFX[2]=VFX[1];
        if(MSP>1)
        {
            for(N=1;N<=MSP;N++)
            {
                get(FSP ,N,2)=get(FSP ,N,1);
            }
        }
        else
            get(FSP ,1,2)=1;
    }

    cout << "READ_DATA functin finished . . . " ;
}   

void HARD_SPHERE()
{
    ////GAS gas;
    ////CALC calc;
    cout<<"Reading HARD_SPHERE Data"<<endl;
    MSP=1;
    MMRM=0;
    MMVM=0;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=1;
    MVIBL=0;
    
    ALLOCATE_GAS();
    
    get(SP ,1,1)=4.0e-10;    //reference diameter
    get(SP ,2,1)=273.0;       //reference temperature
    get(SP ,3,1)=0.5;        //viscosity-temperature index
    get(SP ,4,1)=1.0;         //reciprocal of VSS scattering parameter (1 for VHS)
    get(SP ,5,1)=5.e-26;     //mass
    get(ISPR ,1,1)=0;        //number of rotational degrees of freedom
    cout<<"Hard Sphere data done"<<endl;
    return;
}


void ARGON()
{
    // //GAS gas;
    // //CALC calc;
    cout<<"Reading Argon Data"<<endl;
    MSP=1;
    MMRM=0;
    MMVM=0;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=1;
    MVIBL=0;
    ALLOCATE_GAS();
    get(SP ,1,1)=4.17e-10;
    get(SP ,2,1)=273.15;
    get(SP ,3,1)=0.81;
    get(SP ,4,1)=1.0;
    get(SP ,5,1)=6.63e-26;
    get(ISPR ,1,1)=0;
    get(ISPR ,2,1)=0;
    cout<<"Argon Data done"<<endl;
    return;
}
//
void IDEAL_NITROGEN()
{
    // //GAS gas;
    // //CALC calc;
    cout<<"Reading IDEAL_NITROGEN data"<<endl;
    MSP=1;
    MMRM=1;
    MMVM=0;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=0;
    MVIBL=0;
    //cout << "initaial values set\n" ; // dsuedit
    ALLOCATE_GAS();
    
    //cout << "allocation finished \n" ; // dsuedit
    get(SP ,1,1)=4.17e-10;
    get(SP ,2,1)=273.0;
    get(SP ,3,1)=0.74;
    
    get(SP ,4,1)=1.0;
    get(SP ,5,1)=4.65e-26;
    get(ISPR ,1,1)=2;
    get(ISPR ,2,1)=0;

    get(SPR ,1,1)=5.0;
    cout << "ideal_nitrogen data done\n" ;
    return;
}
//
void REAL_OXYGEN()
{
    //
    //GAS gas;
    //CALC calc;
    cout<<"Reading Real_Oxygen data"<<endl;
    MSP=2;
    MMRM=1;
    MMVM=1;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=5;
    MVIBL=26;
    ALLOCATE_GAS();
    get(SP ,1,1)=4.07e-10;
    get(SP ,2,1)=273.00;
    get(SP ,3,1)=0.77e00;
    get(SP ,4,1)=1.e00;
    get(SP ,5,1)=5.312e-26;
    get(SP ,6,1)=0.e00;
    get(ISPR ,1,1)=2;
    get(ISPR ,2,1)=0 ;            //0,1 for constant,polynomial rotational relaxation collision number
    get(SPR ,1,1)=5.0;             // the collision number or the coefficient of temperature in the polynomial (if a polynomial, the coeff. of T^2 is in spr_db(3  )
    
    get(ISPV  ,1)=1   ;            // the number of vibrational modes
    get(SPVM ,1,1,1)=2256.e00  ;        // the characteristic vibrational temperature
    get(SPVM ,2,1,1)=90000.e00;        // a constant Zv, or the reference Zv
    get(SPVM ,3,1,1)=2256.e00;        // -1 for a constant Zv, or the reference temperature
    get(SPVM ,5,1,1)=1.0;            //arbitrary reduction factor
    get( ISPVM ,1,1,1)=2;
    get( ISPVM ,2,1,1)=2;
    get(NELL  ,1)=3;
    if(MELE > 1) {
        //*
        get(QELC ,1,1,1)=3.0;
        get(QELC ,2,1,1)=0.0;
        get(QELC ,3,1,1)=50.0;  //500.
        get(QELC ,1,2,1)=2.0;
        get(QELC ,2,2,1)=11393.0;
        get(QELC ,3,2,1)=50.0;  //500         //for equipartition, the cross-section ratios must be the same for all levels
        get(QELC ,1,3,1)=1.0;
        get(QELC ,3,3,1)=50.0;  //500.
    }
    //
    //species 2 is atomic oxygen
    get(SP ,1,2)=3.e-10;
    get(SP ,2,2)=273.e00;
    get(SP ,3,2)=0.8e00;
    get(SP ,4,2)=1.e00;
    get(SP ,5,2)=2.656e-26;
    get(SP ,6,2)=4.099e-19;
    get(ISPR ,1,2)=0;
    get(ISPV  ,2)=0;     //must be set//
    //set electronic information
    if(MELE > 1){
        get(NELL  ,2)=5;
        get(QELC ,1,1,2)=5.0;
        get(QELC ,2,1,2)=0.0;
        get(QELC ,2,3,1)=18985.0;
        get(QELC ,3,1,2)=50.0;
        get(QELC ,1,2,2)=3.0;
        get(QELC ,2,2,2)=228.9;
        get(QELC ,3,2,2)=50.0;
        get(QELC ,1,3,2)=1.0;
        get(QELC ,2,3,2)=325.9;
        get(QELC ,3,3,2)=50.0;
        get(QELC ,1,4,2)=5.0;
        get(QELC ,2,4,2)=22830.0;
        get(QELC ,3,4,2)=50.0;
        get(QELC ,1,5,2)=1.0;
        get(QELC ,2,5,2)=48621.0;
        get(QELC ,3,5,2)=50.0;
    }
    //set data needed for recombination
    //
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(ISPRC ,i,j)=0;
            get(ISPRK ,i,j)=0;
        }
    }
    // ISPRC=0;
    // ISPRK=0;
    get(ISPRC ,2,2)=1;    //O+O -> O2  recombined species code for an O+O recombination
    get(ISPRK ,2,2)=1 ;     //the relevant vibrational mode of this species
    get( SPRC ,1,2,2,1)=0.04;
    get( SPRC ,2,2,2,1)=-1.3;
    get( SPRC ,1,2,2,2)=0.05;
    get( SPRC ,2,2,2,2)=-1.1;
    get( SPRT ,1,2,2)=5000.e00;
    get( SPRT ,2,2,2)=15000.e00;
    //
    //memget(NSPEX,0,sizeof(*NSPEX));
    //memget(SPEX,0.e00,sizeof(*SPEX));
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(NSPEX  ,i,j)=0;
        }
    }
    for(int i=1;i<7;i++){
        for(int j=1;j<MMEX+1;j++){
            for(int k=1;k<MSP+1;k++){
                for(int l=1;l<MSP+1;l++)
                    get(SPEX  ,i,j,k,l)=0.e00;
            }
        }
    }
    //SPEX=0.e00;
    //ISPEX=0;
    //
    DERIVED_GAS_DATA();
    //
    cout<<"Real_Oxygen data done"<<endl;
    return;
}
//
void IDEAL_AIR()
{
    //GAS gas;
    //CALC calc;
    cout<<"Reading IDEAL_AIR data"<<endl;
    MSP=2;
    MMRM=1;
    MMVM=0;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=1;
    MVIBL=0;
    //
    ALLOCATE_GAS();
    //
    get(SP ,1,1)=4.07e-10;
    get(SP ,2,1)=273.0;
    get(SP ,3,1)=0.77;
    get(SP ,4,1)=1.0;
    get(SP ,5,1)=5.312e-26;
    get(ISPR ,1,1)=2;
    get(ISPR ,2,1)=0;
    get(SPR ,1,1)=5.0;
    get(SP ,1,2)=4.17e-10;
    get(SP ,2,2)=273.0;
    get(SP ,3,2)=0.74;
    get(SP ,4,2)=1.0;
    get(SP ,5,2)=4.65e-26;
    get(ISPR ,1,2)=2;
    get(ISPR ,2,2)=0;
    get(SPR ,1,2)=5.0;
    cout<<"IDEAL_AIR data done"<<endl;
    return;
}
//
void REAL_AIR()
{
    //GAS gas;
    //CALC calc;
    cout<<"REAL_AIR data done"<<endl;
    MSP=5;
    MMRM=1;
    MMVM=1;
    MELE=5;
    MVIBL=40;  //?
    //
    MEX=4;
    MMEX=1;
    //
    MNSR=0;
    ALLOCATE_GAS();
    //species 1 is oxygen
    get(SP ,1,1)=4.07e-10;
    get(SP ,2,1)=273.e00;
    get(SP ,3,1)=0.77e00;
    get(SP ,4,1)=1.e00;
    get(SP ,5,1)=5.312e-26;
    get(SP ,6,1)=0.e00;
    get(ISPR ,1,1)=2;
    get(ISPR ,2,1)=0;
    get(SPR ,1,1)=5.e00;
    get(ISPV  ,1)=1;               // the number of vibrational modes
    get(SPVM ,1,1,1)=2256.e00;          // the characteristic vibrational temperature
    get(SPVM ,2,1,1)=18000.e00;  //90000.D00        // a constant Zv, or the reference Zv
    get(SPVM ,3,1,1)=2256.e00;       // -1 for a constant Zv, or the reference temperature
    get(SPVM ,5,1,1)=1.0;
    get( ISPVM ,1,1,1)=3;
    get( ISPVM ,2,1,1)=3;
    get(NELL  ,1)=3;
    get(QELC ,1,1,1)=3.0;
    get(QELC ,2,1,1)=0.0;
    get(QELC ,3,1,1)=50.0;
    get(QELC ,1,2,1)=2.0;
    get(QELC ,2,2,1)=11393.0;
    get(QELC ,3,2,1)=50.0;
    get(QELC ,1,3,1)=1.0;
    
    get(QELC ,2,3,1)=18985.0;
    get(QELC ,3,3,1)=50.0;
    //species 2 is nitrogen
    get(SP ,1,2)=4.17e-10;
    get(SP ,2,2)=273.e00;
    get(SP ,3,2)=0.74e00;
    get(SP ,4,2)=1.e00;
    get(SP ,5,2)=4.65e-26;
    get(SP ,6,2)=0.e00;
    get(ISPR ,1,2)=2;
    get(ISPR ,2,2)=0;
    get(SPR ,1,2)=5.e00;
    get(ISPV  ,2)=1;
    get(SPVM ,1,1,2)=3371.e00;
    get(SPVM ,2,1,2)=52000.e00;     //260000.D00
    get(SPVM ,3,1,2)=3371.e00;
    get(SPVM ,5,1,2)=0.3;
    get( ISPVM ,1,1,2)=4;
    get( ISPVM ,2,1,2)=4;
    get(NELL  ,2)=1;
    get(QELC ,1,1,2)=1.0;
    get(QELC ,2,1,2)=0.0;
    get(QELC ,3,1,2)=100.0;
    //species 3 is atomic oxygen
    get(SP ,1,3)=3.e-10;
    get(SP ,2,3)=273.e00;
    get(SP ,3,3)=0.8e00;
    get(SP ,4,3)=1.e00;
    get(SP ,5,3)=2.656e-26;
    get(SP ,6,3)=4.099e-19;
    get(ISPR ,1,3)=0;
    get(ISPV  ,3)=0;
    get(NELL  ,3)=5;
    get(QELC ,1,1,3)=5.0;
    get(QELC ,2,1,3)=0.0;
    get(QELC ,3,1,3)=50.0;
    get(QELC ,1,2,3)=3.0;
    get(QELC ,2,2,3)=228.9;
    get(QELC ,3,2,3)=50.0;
    get(QELC ,1,3,3)=1.0;
    get(QELC ,2,3,3)=325.9;
    get(QELC ,3,3,3)=50.0;
    get(QELC ,1,4,3)=5.0;
    get(QELC ,2,4,3)=22830.0;
    get(QELC ,3,4,3)=50.0;
    get(QELC ,1,5,3)=1.0;
    get(QELC ,2,5,3)=48621.0;
    get(QELC ,3,5,3)=50.0;
    //species 4 is atomic nitrogen
    get(SP ,1,4)=3.e-10;
    get(SP ,2,4)=273.e00;
    get(SP ,3,4)=0.8e00;
    get(SP ,4,4)=1.0e00;
    get(SP ,5,4)=2.325e-26;
    get(SP ,6,4)=7.849e-19;
    get(ISPR ,1,4)=0;
    get(ISPV  ,4)=0;
    get(NELL  ,4)=3;
    get(QELC ,1,1,4)=4.0;
    get(QELC ,2,1,4)=0.0;
    get(QELC ,3,1,4)=50.0;
    get(QELC ,1,2,4)=10.0;
    get(QELC ,2,2,4)=27658.0;
    get(QELC ,3,2,4)=50.0;
    get(QELC ,1,3,4)=6.0;
    get(QELC ,2,3,4)=41495.0;
    get(QELC ,3,3,4)=50.0;
    //species 5 is NO
    get(SP ,1,5)=4.2e-10;
    get(SP ,2,5)=273.e00;
    get(SP ,3,5)=0.79e00;
    get(SP ,4,5)=1.0e00;
    get(SP ,5,5)=4.98e-26;
    get(SP ,6,5)=1.512e-19;
    get(ISPR ,1,5)=2;
    get(ISPR ,2,5)=0;
    get(SPR ,1,5)=5.e00;
    get(ISPV  ,5)=1;
    get(SPVM ,1,1,5)=2719.e00;
    get(SPVM ,2,1,5)=14000.e00;   //70000.D00
    get(SPVM ,3,1,5)=2719.e00;
    get(SPVM ,5,1,5)=0.2;
    get( ISPVM ,1,1,5)=3;
    get( ISPVM ,2,1,5)=4;
    get(NELL  ,5)=2;
    get(QELC ,1,1,5)=2.0;
    get(QELC ,2,1,5)=0.0;
    get(QELC ,3,1,5)=50.0;
    get(QELC ,1,2,5)=2.0;
    get(QELC ,2,2,5)=174.2;
    get(QELC ,3,2,5)=50.0;
    //set the recombination data for the molecule pairs
    //memget(ISPRC,0,sizeof(*ISPRC));//ISPRC=0;    //data os zero unless explicitly set
    //memget(ISPRK,0,sizeof(*ISPRK));//ISPRK=0;
    //memget(SPRC,0,sizeof(*SPRC));//SPRC=0.e00;
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(ISPRC ,i,j)=0;
        }
    }
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(ISPRK ,i,j)=0;
        }
    }
    for(int i=1;i<5;i++){
        for(int j=1;j<MSP+1;j++){
            for(int k=1;k<MSP+1;k++){
                for(int l=1;l<MSP+1;l++)
                    get(SPEX  ,i,j,k,l)=0.e00;
            }
        }
    }
    get(ISPRC ,3,3)=1; //O+O -> O2  recombined species code for an O+O recombination
    get(ISPRK ,3,3)=1;
    get( SPRC ,1,3,3,1)=0.04e00;
    get( SPRC ,2,3,3,1)=-1.3e00;
    get( SPRC ,1,3,3,2)=0.07e00;
    get( SPRC ,2,3,3,2)=-1.2e00;
    get( SPRC ,1,3,3,3)=0.08e00;
    get( SPRC ,2,3,3,3)=-1.2e00;
    get( SPRC ,1,3,3,4)=0.09e00;
    get( SPRC ,2,3,3,4)=-1.2e00;
    get( SPRC ,1,3,3,5)=0.065e00;
    get( SPRC ,2,3,3,5)=-1.2e00;
    get( SPRT ,1,3,3)=5000.e00;
    get( SPRT ,2,3,3)=15000.e00;
    get(ISPRC ,4,4)=2;  //N+N -> N2
    get(ISPRK ,4,4)=1;
    get( SPRC ,1,4,4,1)=0.15e00;
    get( SPRC ,2,4,4,1)=-2.05e00;
    get( SPRC ,1,4,4,2)=0.09e00;
    get( SPRC ,2,4,4,2)=-2.1e00;
    get( SPRC ,1,4,4,3)=0.16e00;
    get( SPRC ,2,4,4,3)=-2.0e00;
    get( SPRC ,1,4,4,4)=0.17e00;
    get( SPRC ,2,4,4,4)=-2.0e00;
    get( SPRC ,1,4,4,5)=0.17e00;
    get( SPRC ,2,4,4,5)=-2.1e00;
    get( SPRT ,1,4,4)=5000.e00;
    get( SPRT ,2,4,4)=15000.e00;
    get(ISPRC ,3,4)=5;
    get(ISPRK ,3,4)=1;
    get( SPRC ,1,3,4,1)=0.3e00;
    get( SPRC ,2,3,4,1)=-1.9e00;
    get( SPRC ,1,3,4,2)=0.4e00;
    get( SPRC ,2,3,4,2)=-2.0e00;
    get( SPRC ,1,3,4,3)=0.3e00;
    get( SPRC ,2,3,4,3)=-1.75e00;
    get( SPRC ,1,3,4,4)=0.3e00;
    get( SPRC ,2,3,4,4)=-1.75e00;
    get( SPRC ,1,3,4,5)=0.15e00;
    get( SPRC ,2,3,4,5)=-1.9e00;
    get( SPRT ,1,3,4)=5000.e00;
    get( SPRT ,2,3,4)=15000.e00;
    //set the exchange reaction data
    //memget(SPEX,0,sizeof(*SPEX));//SPEX=0.e00;
    for(int i=1;i<7;i++){
        for(int j=1;j<MMEX+1;j++){
            for(int k=1;k<MSP+1;k++){
                for(int l=1;l<MSP+1;l++)
                    get(SPEX  ,i,j,k,l)=0.e00;
            }
        }
    }
    //ISPEX=0;
    //NSPEX=0;
    get(NSPEX  ,2,3)=1;
    get(NSPEX  ,4,5)=1;
    get(NSPEX  ,3,5)=1;
    get(NSPEX  ,1,4)=1;
    //N2+O->NO+N
    get(ISPEX  ,1,1,2,3)=2;
    get(ISPEX  ,1,2,2,3)=3;
    get(ISPEX  ,1,3,2,3)=5;
    get(ISPEX  ,1,4,2,3)=4;
    get(ISPEX  ,1,5,2,3)=1;
    get(ISPEX  ,1,6,2,3)=1;
    get(SPEX  ,6,1,2,3)=0.e00;
    get(NEX  ,1,2,3)=1;
    //NO+N->N2+0
    get(ISPEX  ,1,1,4,5)=5;
    get(ISPEX  ,1,2,4,5)=4;
    get(ISPEX  ,1,3,4,5)=2;
    get(ISPEX  ,1,4,4,5)=3;
    get(ISPEX  ,1,5,4,5)=1;
    get(ISPEX  ,1,6,4,5)=1;
    get(ISPEX  ,1,7,4,5)=1;
    get(SPEX  ,1,1,4,5)=0.8e00;
    get(SPEX  ,2,1,4,5)=-0.75e00;
    get(SPEX  ,4,1,4,5)=5000.e00;
    get(SPEX  ,5,1,4,5)=15000.e00;
    get(SPEX  ,6,1,4,5)=0.e00;
    get(NEX  ,1,4,5)=2;
    //NO+O->O2+N
    get(ISPEX  ,1,1,3,5)=5;
    get(ISPEX  ,1,2,3,5)=3;
    get(ISPEX  ,1,3,3,5)=1;
    get(ISPEX  ,1,4,3,5)=4;
    get(ISPEX  ,1,5,3,5)=1;
    get(ISPEX  ,1,6,3,5)=1;
    get(SPEX  ,6,1,3,5)=2.e-19;
    get(NEX  ,1,3,5)=3;
    //O2+N->NO+O
    get(ISPEX  ,1,1,1,4)=1;
    get(ISPEX  ,1,2,1,4)=4;
    get(ISPEX  ,1,3,1,4)=5;
    get(ISPEX  ,1,4,1,4)=3;
    get(ISPEX  ,1,5,1,4)=1;
    get(ISPEX  ,1,6,1,4)=1;
    get(ISPEX  ,1,7,1,4)=1 ;
    get(SPEX  ,1,1,1,4)=7.e00;
    get(SPEX  ,2,1,1,4)=-0.85e00;
    get(SPEX  ,4,1,1,4)=5000.e00;
    get(SPEX  ,5,1,1,4)=15000.e00;
    get(SPEX  ,6,1,1,4)=0.e00;
    get(NEX  ,1,1,4)=4;
    
    DERIVED_GAS_DATA();
    cout<<"REAL_AIR data done"<<endl;
    return;
}
//
void HELIUM_ARGON_XENON()
{
    //GAS gas;
    //CALC calc;
    cout<<"Reading HELIUM_ARGON_XENON data"<<endl;
    MSP=3;
    MMRM=0;
    MMVM=0;
    MNSR=0;
    MEX=0;
    MMEX=0;
    MELE=1;
    MVIBL=0;
    
    ALLOCATE_GAS();
    
    get(SP ,1,1)=2.30e-10;   //2.33D-10
    get(SP ,2,1)=273.0;
    get(SP ,3,1)=0.66;
    get(SP ,4,1)=0.794;   //1.
    get(SP ,5,1)=6.65e-27;
    get(ISPR ,1,1)=0;
    get(ISPR ,2,1)=0;
    //
    get(SP ,1,2)=4.11e-10;   //4.17D-10
    get(SP ,2,2)=273.15;
    get(SP ,3,2)=0.81;
    get(SP ,4,2)=0.714;    //1.
    get(SP ,5,2)=6.63e-26;
    get(ISPR ,1,2)=0;
    get(ISPR ,2,2)=0;
    //
    get(SP ,1,3)=5.65e-10;   //5.74D-10
    get(SP ,2,3)=273.0;
    get(SP ,3,3)=0.85;
    get(SP ,4,3)=0.694;   //1.
    get(SP ,5,3)=21.8e-26;
    get(ISPR ,1,3)=0;
    get(ISPR ,2,3)=0;
    cout<<"HELIUM_ARGON_XENON data done"<<endl;
    return;
}
//
void OXYGEN_HYDROGEN()
{
    //
    //GAS gas;
    //CALC calc;
    cout<<"Reading OXYGEN_HYDROGEN data"<<endl;
    MSP=8;
    MMRM=3;
    MMVM=3;
    MELE=1;
    MVIBL=40;  //the maximum number of vibrational levels before a cumulative level reaches 1
    //
    MEX=16;
    MMEX=3;
    //
    MNSR=0;
    //
    ALLOCATE_GAS();
    //
    //species 1 is hydrogen H2
    get(SP ,1,1)=2.92e-10;
    get(SP ,2,1)=273.e00;
    get(SP ,3,1)=0.67e00;
    get(SP ,4,1)=1.e00;
    get(SP ,5,1)=3.34e-27;
    get(SP ,6,1)=0.e00;
    get(ISPR ,1,1)=2;
    get(ISPR ,2,1)=0;
    get(SPR ,1,1)=5.e00;
    get(ISPV  ,1)=1;         // the number of vibrational modes
    get(SPVM ,1,1,1)=6159.e00;          // the characteristic vibrational temperature
    get(SPVM ,2,1,1)=20000.e00;  //estimate
    get(SPVM ,3,1,1)=2000.e00; //estimate
    get(SPVM ,5,1,1)=1.0;
    get( ISPVM ,1,1,1)=2;
    get( ISPVM ,2,1,1)=2;
    //species 2 is atomic hydrogen H
    get(SP ,1,2)=2.5e-10;      //estimate
    get(SP ,2,2)=273.e00;
    get(SP ,3,2)=0.8e00;
    get(SP ,4,2)=1.e00;
    get(SP ,5,2)=1.67e-27;
    get(SP ,6,2)=3.62e-19;
    get(ISPR ,1,2)=0;
    get(ISPV  ,2)=0;
    //species 3 is oxygen O2
    get(SP ,1,3)=4.07e-10;
    get(SP ,2,3)=273.e00;
    get(SP ,3,3)=0.77e00;
    get(SP ,4,3)=1.e00;
    get(SP ,5,3)=5.312e-26;
    get(SP ,6,3)=0.e00;
    get(ISPR ,1,3)=2;
    get(ISPR ,2,3)=0;
    get(SPR ,1,3)=5.e00;
    get(ISPV  ,3)=1;               // the number of vibrational modes
    get(SPVM ,1,1,3)=2256.e00;          // the characteristic vibrational temperature
    get(SPVM ,2,1,3)=18000.e00;  //90000.D00        // a constant Zv, or the reference Zv
    get(SPVM ,3,1,3)=2256.e00;       // -1 for a constant Zv, or the reference temperature
    get(SPVM ,5,1,3)=1.e00;
    get( ISPVM ,1,1,3)=4;
    get( ISPVM ,2,1,3)=4;
    //species 4 is atomic oxygen O
    get(SP ,1,4)=3.e-10;    //estimate
    get(SP ,2,4)=273.e00;
    get(SP ,3,4)=0.8e00;
    get(SP ,4,4)=1.e00;
    get(SP ,5,4)=2.656e-26;
    get(SP ,6,4)=4.099e-19;
    get(ISPR ,1,4)=0;
    get(ISPV  ,4)=0;
    //species 5 is hydroxy OH
    get(SP ,1,5)=4.e-10;       //estimate
    get(SP ,2,5)=273.e00;
    get(SP ,3,5)=0.75e00;      //-estimate
    get(SP ,4,5)=1.0e00;
    get(SP ,5,5)=2.823e-26;
    get(SP ,6,5)=6.204e-20;
    get(ISPR ,1,5)=2;
    get(ISPR ,2,5)=0;
    get(SPR ,1,5)=5.e00;
    get(ISPV  ,5)=1;
    get(SPVM ,1,1,5)=5360.e00;
    get(SPVM ,2,1,5)=20000.e00;   //estimate
    get(SPVM ,3,1,5)=2500.e00;    //estimate
    get(SPVM ,5,1,5)=1.0e00;
    get( ISPVM ,1,1,5)=2;
    get( ISPVM ,2,1,5)=4;
    //species 6 is water vapor H2O
    get(SP ,1,6)=4.5e-10;      //estimate
    get(SP ,2,6)=273.e00;
    get(SP ,3,6)=0.75e00 ;     //-estimate
    get(SP ,4,6)=1.0e00;
    get(SP ,5,6)=2.99e-26;
    get(SP ,6,6)=-4.015e-19;
    get(ISPR ,1,6)=3;
    get(ISPR ,2,6)=0;
    get(SPR ,1,6)=5.e00;
    get(ISPV  ,6)=3;
    get(SPVM ,1,1,6)=5261.e00;  //symmetric stretch mode
    get(SPVM ,2,1,6)=20000.e00;   //estimate
    get(SPVM ,3,1,6)=2500.e00;    //estimate
    get(SPVM ,5,1,6)=1.e00;
    get(SPVM ,1,2,6)=2294.e00;  //bend mode
    get(SPVM ,2,2,6)=20000.e00;   //estimate
    get(SPVM ,3,2,6)=2500.e00;    //estimate
    get(SPVM ,5,2,6)=1.0e00;
    get(SPVM ,1,3,6)=5432.e00;  //asymmetric stretch mode
    get(SPVM ,2,3,6)=20000.e00;   //estimate
    get(SPVM ,3,3,6)=2500.e00 ;   //estimate
    get(SPVM ,5,3,6)=1.e00;
    get( ISPVM ,1,1,6)=2;
    get( ISPVM ,2,1,6)=5;
    get( ISPVM ,1,2,6)=2;
    get( ISPVM ,2,2,6)=5;
    get( ISPVM ,1,3,6)=2;
    get( ISPVM ,2,3,6)=5;
    //species 7 is hydroperoxy HO2
    get(SP ,1,7)=5.5e-10;       //estimate
    get(SP ,2,7)=273.e00;
    get(SP ,3,7)=0.75e00 ;     //-estimate
    get(SP ,4,7)=1.0e00;
    get(SP ,5,7)=5.479e-26;
    get(SP ,6,7)=2.04e-20;
    get(ISPR ,1,7)=2;    //assumes that HO2 is linear
    get(ISPR ,2,7)=0;
    get(SPR ,1,7)=5.e00;
    get(ISPV  ,7)=3;
    get(SPVM ,1,1,7)=4950.e00;
    get(SPVM ,2,1,7)=20000.e00;   //estimate
    get(SPVM ,3,1,7)=2500.e00  ;  //estimate
    get(SPVM ,5,1,7)=1.e00;
    get(SPVM ,1,2,7)=2000.e00;
    get(SPVM ,2,2,7)=20000.e00;   //estimate
    get(SPVM ,3,2,7)=2500.e00;    //estimate
    get(SPVM ,5,2,7)=1.e00;
    get(SPVM ,1,3,7)=1580.e00;
    get(SPVM ,2,3,7)=20000.e00;   //estimate
    get(SPVM ,3,3,7)=2500.e00;    //estimate
    get(SPVM ,5,3,7)=1.e00;
    get( ISPVM ,1,1,7)=2;
    get( ISPVM ,2,1,7)=3;
    get( ISPVM ,1,2,7)=2;
    get( ISPVM ,2,2,7)=3;
    get( ISPVM ,1,3,7)=2;
    get( ISPVM ,2,3,7)=3;
    //Species 8 is argon
    get(SP ,1,8)=4.17e-10;
    get(SP ,2,8)=273.15;
    get(SP ,3,8)=0.81   ;
    get(SP ,4,8)=1.0;
    get(SP ,5,8)=6.63e-26;
    get(SP ,6,8)=0.e00;
    get(ISPR ,1,8)=0;
    get(ISPV  ,8)=0;
    //
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(ISPRC ,i,j)=0;
        }
    }
    //ISPRC=0;    //data is zero unless explicitly set
    //
    get(ISPRC ,4,4)=3;    //O+O+M -> O2+M  recombined species code for an O+O recombination
    get(ISPRK ,4,4)=1;
    get( SPRC ,1,4,4,1)=0.26e00;
    get( SPRC ,2,4,4,1)=-1.3e00;
    get( SPRC ,1,4,4,2)=0.29e00;
    get( SPRC ,2,4,4,2)=-1.3e00;
    get( SPRC ,1,4,4,3)=0.04e00;
    get( SPRC ,2,4,4,3)=-1.5e00;
    get( SPRC ,1,4,4,4)=0.1e00;
    get( SPRC ,2,4,4,4)=-1.4e00;
    get( SPRC ,1,4,4,5)=0.1e00;
    get( SPRC ,2,4,4,5)=-1.4e00;
    get( SPRC ,1,4,4,6)=0.1e00;
    get( SPRC ,2,4,4,6)=-1.4e00;
    get( SPRC ,1,4,4,7)=0.07e00;
    get( SPRC ,2,4,4,7)=-1.5e00;
    get( SPRC ,1,4,4,8)=0.07e00;
    get( SPRC ,2,4,4,8)=-1.5e00;
    get( SPRT ,1,4,4)=1000.e00;
    get( SPRT ,2,4,4)=3000.e00;
    //
    get(ISPRC ,2,2)=1;   //H+H+M -> H2+M
    get(ISPRK ,2,2)=1;
    get( SPRC ,1,2,2,1)=0.07e00;
    get( SPRC ,2,2,2,1)=-2.e00;
    get( SPRC ,1,2,2,2)=0.11e00;
    get( SPRC ,2,2,2,2)=-2.2e00;
    get( SPRC ,1,2,2,3)=0.052e00;
    get( SPRC ,2,2,2,3)=-2.5e00;
    get( SPRC ,1,2,2,4)=0.052e00;
    get( SPRC ,2,2,2,4)=-2.5e00;
    get( SPRC ,1,2,2,5)=0.052e00;
    get( SPRC ,2,2,2,5)=-2.5e00;
    get( SPRC ,1,2,2,6)=0.052e00;
    get( SPRC ,2,2,2,6)=-2.5e00;
    get( SPRC ,1,2,2,7)=0.052e00;
    get( SPRC ,2,2,2,7)=-2.5e00;
    get( SPRC ,1,2,2,8)=0.04e00;
    get( SPRC ,2,2,2,7)=-2.5e00;
    get( SPRT ,1,2,2)=1000.e00;
    get( SPRT ,2,2,2)=3000.e00;
    //
    get(ISPRC ,2,4)=5;    //H+0+M -> OH+M
    get(ISPRK ,2,4)=1;
    get( SPRC ,1,2,4,1)=0.15e00;
    get( SPRC ,2,2,4,1)=-2.e00;
    get( SPRC ,1,2,4,2)=0.04e00;
    get( SPRC ,2,2,4,2)=-1.3e00;
    get( SPRC ,1,2,4,3)=0.04e00;
    get( SPRC ,2,2,4,3)=-1.3e00;
    get( SPRC ,1,2,4,4)=0.04e00;
    get( SPRC ,2,2,4,4)=-1.3e00;
    get( SPRC ,1,2,4,5)=0.04e00;
    get( SPRC ,2,2,4,5)=-1.3e00;
    get( SPRC ,1,2,4,6)=0.21e00;
    get( SPRC ,2,2,4,6)=-2.1e00;
    get( SPRC ,1,2,4,7)=0.18e00;
    get( SPRC ,2,2,4,7)=-2.3e00;
    get( SPRC ,1,2,4,8)=0.16e00;
    get( SPRC ,2,2,4,8)=-2.3e00;
    get( SPRT ,1,2,4)=1000.e00;
    get( SPRT ,2,2,4)=3000.e00;
    //
    get(ISPRC ,2,5)=6;    //H+OH+M -> H2O+M
    get(ISPRK ,2,5)=1;
    get( SPRC ,1,2,5,1)=0.1e00;
    get( SPRC ,2,2,5,1)=-2.0e00;
    get( SPRC ,1,2,5,2)=0.1e00;
    get( SPRC ,2,2,5,2)=-2.0e00;
    get( SPRC ,1,2,5,3)=0.0025e00;
    get( SPRC ,2,2,5,3)=-2.2e00;
    get( SPRC ,1,2,5,4)=0.0025e00;
    get( SPRC ,2,2,5,4)=-2.2e00;
    get( SPRC ,1,2,5,5)=0.0025e00;
    get( SPRC ,2,2,5,5)=-2.2e00;
    get( SPRC ,1,2,5,6)=0.0015e00;
    get( SPRC ,2,2,5,6)=-2.2e00;
    get( SPRC ,1,2,5,7)=0.0027e00;
    get( SPRC ,2,2,5,7)=-2.e00;
    get( SPRC ,1,2,5,8)=0.0025e00;
    get( SPRC ,2,2,5,8)=-2.e00;
    get( SPRT ,1,2,5)=1000.e00;
    get( SPRT ,2,2,5)=3000.e00;
    //
    get(ISPRC ,2,3)=7;   //H+O2+M -> H02+M
    get(ISPRK ,2,3)=1;
    get( SPRC ,1,2,3,1)=0.0001e00;
    get( SPRC ,2,2,3,1)=-1.7e00;
    get( SPRC ,1,2,3,2)=0.0001e00;
    get( SPRC ,2,2,3,2)=-1.7e00;
    get( SPRC ,1,2,3,3)=0.00003e00;
    get( SPRC ,2,2,3,3)=-1.5e00;
    get( SPRC ,1,2,3,4)=0.00003e00;
    get( SPRC ,2,2,3,4)=-1.7e00;
    get( SPRC ,1,2,3,5)=0.00003e00;
    get( SPRC ,2,2,3,5)=-1.7e00;
    get( SPRC ,1,2,3,6)=0.00003e00;
    get( SPRC ,2,2,3,6)=-1.7e00;
    get( SPRC ,1,2,3,7)=0.000012e00;
    get( SPRC ,2,2,3,7)=-1.7e00;
    get( SPRC ,1,2,3,8)=0.00002e00;
    get( SPRC ,2,2,3,8)=-1.7e00;
    get( SPRT ,1,2,3)=1000.e00;
    get( SPRT ,2,2,3)=3000.e00;
    //
    //set the exchange reaction data
    //  memget(SPEX,0,sizeof(*SPEX));//SPEX=0.e00;    //all activation energies and heats of reaction are zero unless set otherwise
    for(int i=1;i<7;i++){
        for(int j=1;j<MMEX+1;j++){
            for(int k=1;k<MSP+1;k++){
                for(int l=1;l<MSP+1;l++)
                    get(SPEX  ,i,j,k,l)=0.e00;
            }
        }
    }
    //ISPEX=0;       // ISPEX is also zero unless set otherwise
    for(int i=1;i<MMEX+1;i++){
        for(int j=1;j<8;j++){
            for(int k=1;k<MSP+1;k++){
                for(int l=1;l<MSP+1;l++)
                    get(ISPEX  ,i,j,k,l)=0.e00;
            }
        }
    }
    //NSPEX=0;
    for(int i=1;i<MSP+1;i++){
        for(int j=1;j<MSP+1;j++){
            get(NSPEX  ,i,j)=0;
        }
    }
    //set the number of exchange reactions for each species pair
    get(NSPEX  ,1,3)=1;
    get(NSPEX  ,2,7)=3;
    get(NSPEX  ,2,3)=1;
    get(NSPEX  ,4,5)=1;
    get(NSPEX  ,1,4)=1;
    get(NSPEX  ,2,5)=1;
    get(NSPEX  ,1,5)=1;
    get(NSPEX  ,2,6)=1;
    get(NSPEX  ,4,6)=2;
    get(NSPEX  ,5,5)=2;
    get(NSPEX  ,4,7)=1;
    get(NSPEX  ,3,5)=1;
    //set the information on the chain reactions
    //
    //H2+O2 -> HO2+H
    get(ISPEX  ,1,1,1,3)=1;
    get(ISPEX  ,1,2,1,3)=3;
    get(ISPEX  ,1,3,1,3)=7;
    get(ISPEX  ,1,4,1,3)=2;
    get(ISPEX  ,1,5,1,3)=1;
    get(ISPEX  ,1,6,1,3)=1;
    get(SPEX  ,6,1,1,3)=0.e00;
    get(NEX  ,1,1,3)=1;
    //
    //HO2+H -> H2+02
    get(ISPEX  ,1,1,2,7)=7;
    get(ISPEX  ,1,2,2,7)=2;
    get(ISPEX  ,1,3,2,7)=1;
    get(ISPEX  ,1,4,2,7)=3;
    get(ISPEX  ,1,5,2,7)=1;
    get(ISPEX  ,1,6,2,7)=1;
    get(ISPEX  ,1,7,2,7)=1;
    //H02 is H-O-O so that not all vibrational modes contribute to this reaction, but the numbers here are guesses//
    get(SPEX  ,1,1,2,7)=20.e00;
    get(SPEX  ,2,1,2,7)=0.4e00;
    get(SPEX  ,4,1,2,7)=2000.e00;
    get(SPEX  ,5,1,2,7)=3000.e00;
    get(SPEX  ,6,1,2,7)=0.e00;
    get(NEX  ,1,2,7)=2;
    //
    //O2+H -> OH+O
    get(ISPEX  ,1,1,2,3)=3;
    get(ISPEX  ,1,2,2,3)=2;
    get(ISPEX  ,1,3,2,3)=5;
    get(ISPEX  ,1,4,2,3)=4;
    get(ISPEX  ,1,5,2,3)=1;
    get(ISPEX  ,1,6,2,3)=1;
    get(SPEX  ,6,1,2,3)=0.e00;
    get(NEX  ,1,2,3)=3;
    //
    //OH+O -> O2+H
    get(ISPEX  ,1,1,4,5)=5;
    get(ISPEX  ,1,2,4,5)=4;
    get(ISPEX  ,1,3,4,5)=3;
    get(ISPEX  ,1,4,4,5)=2;
    get(ISPEX  ,1,5,4,5)=1;
    get(ISPEX  ,1,6,4,5)=1;
    get(ISPEX  ,1,7,4,5)=1;
    get(SPEX  ,1,1,4,5)=0.65e00;
    get(SPEX  ,2,1,4,5)=-0.26;
    get(SPEX  ,4,1,4,5)=2000.e00;
    get(SPEX  ,5,1,4,5)=3000.e00;
    get(SPEX  ,6,1,4,5)=0.e00;
    get(NEX  ,1,4,5)=4;
    //
    //H2+O -> OH+H
    get(ISPEX  ,1,1,1,4)=1;
    get(ISPEX  ,1,2,1,4)=4;
    get(ISPEX  ,1,3,1,4)=5;
    get(ISPEX  ,1,4,1,4)=2;
    get(ISPEX  ,1,5,1,4)=1;
    get(ISPEX  ,1,6,1,4)=1;
    get(SPEX  ,6,1,1,4)=0.e00;
    get(NEX  ,1,1,4)=5;
    //
    //OH+H -> H2+O
    get(ISPEX  ,1,1,2,5)=5;
    get(ISPEX  ,1,2,2,5)=2;
    get(ISPEX  ,1,3,2,5)=1;
    get(ISPEX  ,1,4,2,5)=4;
    get(ISPEX  ,1,5,2,5)=1;
    get(ISPEX  ,1,6,2,5)=1;
    get(ISPEX  ,1,7,2,5)=1;
    get(SPEX  ,1,1,2,5)=0.5e00;
    get(SPEX  ,2,1,2,5)=-0.2e00;
    get(SPEX  ,4,1,2,5)=2000.e00;
    get(SPEX  ,5,1,2,5)=3000.e00;
    get(SPEX  ,6,1,2,5)=0.e00;
    get(NEX  ,1,2,5)=6;
    //
    //H20+H -> OH+H2
    get(ISPEX  ,1,1,2,6)=6;
    get(ISPEX  ,1,2,2,6)=2;
    get(ISPEX  ,1,3,2,6)=5;
    get(ISPEX  ,1,4,2,6)=1;
    get(ISPEX  ,1,5,2,6)=1;
    get(ISPEX  ,1,6,2,6)=1;
    get(SPEX  ,6,1,2,6)=2.0e-19;
    get(NEX  ,1,2,6)=7;
    
    //OH+H2 -> H2O+H
    get(ISPEX  ,1,1,1,5)=5;
    get(ISPEX  ,1,2,1,5)=1;
    get(ISPEX  ,1,3,1,5)=6;
    get(ISPEX  ,1,4,1,5)=2;
    get(ISPEX  ,1,5,1,5)=1;
    get(ISPEX  ,1,6,1,5)=1;
    get(ISPEX  ,1,7,1,5)=1;
    get(SPEX  ,1,1,1,5)=0.5;
    get(SPEX  ,2,1,1,5)=-0.2;
    get(SPEX  ,4,1,1,5)=2000.e00;
    get(SPEX  ,5,1,1,5)=3000.e00;
    get(SPEX  ,6,1,1,5)=0.e00;
    get(NEX  ,1,1,5)=8;
    //
    //H2O+O -> OH+OH
    get(ISPEX  ,1,1,4,6)=6;
    get(ISPEX  ,1,2,4,6)=4;
    get(ISPEX  ,1,3,4,6)=5;
    get(ISPEX  ,1,4,4,6)=5;
    get(ISPEX  ,1,5,4,6)=1;
    get(ISPEX  ,1,6,4,6)=1;
    get(SPEX  ,6,1,4,6)=0.e00;
    get(NEX  ,1,4,6)=9;
    //
    //0H+OH -> H2O+O
    get(ISPEX  ,1,1,5,5)=5;
    get(ISPEX  ,1,2,5,5)=5;
    get(ISPEX  ,1,3,5,5)=6;
    get(ISPEX  ,1,4,5,5)=4;
    get(ISPEX  ,1,5,5,5)=1;
    get(ISPEX  ,1,6,5,5)=1;
    get(ISPEX  ,1,7,5,5)=1;
    get(SPEX  ,1,1,5,5)=0.35;
    get(SPEX  ,2,1,5,5)=-0.2 ;
    get(SPEX  ,4,1,5,5)=2000.e00;
    get(SPEX  ,5,1,5,5)=3000.e00;
    get(SPEX  ,6,1,5,5)=0.e00;
    get(NEX  ,1,5,5)=10;
    //
    //OH+OH  -> HO2+H
    //
    get(ISPEX  ,2,1,5,5)=5;
    get(ISPEX  ,2,2,5,5)=5;
    get(ISPEX  ,2,3,5,5)=7;
    get(ISPEX  ,2,4,5,5)=2;
    get(ISPEX  ,2,5,5,5)=1;
    get(ISPEX  ,2,6,5,5)=1;
    get(SPEX  ,6,2,5,5)=0.e00;
    get(NEX  ,2,5,5)=11;
    //
    //H02+H -> 0H+OH
    get(ISPEX  ,2,1,2,7)=7;
    get(ISPEX  ,2,2,2,7)=2;
    get(ISPEX  ,2,3,2,7)=5;
    get(ISPEX  ,2,4,2,7)=5;
    get(ISPEX  ,2,5,2,7)=1;
    get(ISPEX  ,2,6,2,7)=1;
    get(ISPEX  ,2,7,2,7)=1;
    get(SPEX  ,1,2,2,7)=120.e00;
    get(SPEX  ,2,2,2,7)=-0.05e00;
    get(SPEX  ,4,2,2,7)=2000.e00;
    get(SPEX  ,5,2,2,7)=3000.e00;
    get(SPEX  ,6,2,2,7)=0.e00;
    get(NEX  ,2,2,7)=12;
    //
    //H2O+O -> HO2+H
    //
    get(ISPEX  ,2,1,4,6)=6;
    get(ISPEX  ,2,2,4,6)=4;
    get(ISPEX  ,2,3,4,6)=7;
    get(ISPEX  ,2,4,4,6)=2;
    get(ISPEX  ,2,5,4,6)=1;
    get(ISPEX  ,2,6,4,6)=1;
    get(SPEX  ,6,2,4,6)=0.e00;
    get(NEX  ,2,4,6)=13;
    //
    //H02+H -> H2O+O
    //
    get(ISPEX  ,3,1,2,7)=7;
    get(ISPEX  ,3,2,2,7)=2;
    get(ISPEX  ,3,3,2,7)=6;
    get(ISPEX  ,3,4,2,7)=4;
    get(ISPEX  ,3,5,2,7)=1;
    get(ISPEX  ,3,6,2,7)=1;
    get(ISPEX  ,3,7,2,7)=1;
    get(SPEX  ,1,3,2,7)=40.e00;
    get(SPEX  ,2,3,2,7)=-1.e00;
    get(SPEX  ,4,3,2,7)=2000.e00;
    get(SPEX  ,5,3,2,7)=3000.e00;
    get(SPEX  ,6,3,2,7)=0.e00;
    get(NEX  ,3,2,7)=14;
    //
    //OH+O2 -> HO2+O
    //
    get(ISPEX  ,1,1,3,5)=5;
    get(ISPEX  ,1,2,3,5)=3;
    get(ISPEX  ,1,3,3,5)=7;
    get(ISPEX  ,1,4,3,5)=4;
    get(ISPEX  ,1,5,3,5)=1;
    get(ISPEX  ,1,6,3,5)=1;
    get(SPEX  ,6,1,3,5)=0.e00;
    get(NEX  ,1,3,5)=15;
    //
    //H02+0 -> OH+O2
    //
    get(ISPEX  ,1,1,4,7)=7;
    get(ISPEX  ,1,2,4,7)=4;
    get(ISPEX  ,1,3,4,7)=5;
    get(ISPEX  ,1,4,4,7)=3;
    get(ISPEX  ,1,5,4,7)=1;
    get(ISPEX  ,1,6,4,7)=1;
    get(ISPEX  ,1,7,4,7)=1;
    get(SPEX  ,1,1,4,7)=100.e00;
    get(SPEX  ,2,1,4,7)=0.15e00;
    get(SPEX  ,4,1,4,7)=2000.e00;
    get(SPEX  ,5,1,4,7)=3000.e00;
    get(SPEX  ,6,1,4,7)=0.e00;
    get(NEX  ,1,4,7)=16;
    
    //
    DERIVED_GAS_DATA();
    //
    cout<<"OXYGEN_HYDROGEN data done"<<endl;
    return;
}
//****
//**END OF GAS DATABASE**
//****
//

//module 
int main()
{
    int IRUN,ICONF,N,M,IADAPT,IRETREM,ISET ;
    double A ;

    NVER =1 ;
    MVER =1 ;
    NREL = 1 ;

    //constants
    PI=3.1415926535897932E00 ;
    DPI=6.283185307179586E00 ;
    SPI=1.772453850905516E00 ; 
    BOLTZ=1.380658E-23 ;
    AVOG=6.022169E26     ;

    //adjustable computational parametres
    NMCC = 50 ;
    CPDTM = 0.2 ;
    TPDTM = 0.5 ;
    NNC = 1 ;
    SAMPRAT = 5 ;
    OUTRAT = 10 ;
    FRACSAM =1 ;
    ISAD = 0 ;
    IMTS =2 ;
    FNUMF =1 ;
    TLIM = 1.E20 ;

    
    file_9.open("DIAG.TXT" , ios::out ) ;
    int a ;
    if(file_9.is_open())
    {
        cout << "file_9 DIAG.TXT IS OPEN \n" ;
    }
    else cout << "file_9 DIAG.TXT COULDN'T BE OPEN OPENED" ;

    file_13.open("MolNum.DAT" , ios::out ) ;

    IVB =0 ;

    IRUN=2 ;
    if(IRUN == 1)   cout << "continuing an existing run . . .  \n";
    if(IRUN == 2)
    {
        cout << "enter 0 for a homogenous gas \n" ;
        cout << "Enter 1 for a one-dimensional flow, or\n" ;
        cout << "Enter 2 for a two-dimensional plane flow, or \n" ;
        cout << "Enter 3 for a three dimensional flow, or \n" ;
        cout << "Enter 4 for an axially-symmetric flow :- \n" ;
        ICLASS = 0 ;    // dsuedit
        //cin >> ICLASS ; //dsuedit ICLASS =0
        NCLASS =2 ;

        if (ICLASS < 2)
        {
            
            NCLASS=1 ;
        }
        if(ICLASS == 3) NCLASS =3 ;

        cout << "Enter 0 for an eventually steady flow, or\n" ;
        cout << "enter 1 for a continuing unsteady flow :-\n" ;
        cin >> ISF ;  //dsuedit   assuming steady state all the time .
        //ISF = 0 ;   //dsuedit
        file_9 << "Starting a new run with ICLASS, ISF" <<ICLASS<<ISF << endl ;
    }


    if( IRUN == 2 )
    {
        READ_DATA() ;
        if (ICLASS<2)
        {
            
            SET_INITIAL_STATE_1D() ;
        }
        if(ICLASS==0)   ENERGY(0,A) ;

    }
    int ch = 0 ;

    while(FTIME < TLIM) {
        

        clock_t t[8] ;
        FTIME = FTIME+DTM ;
        file_9 << "TIME " << FTIME << "\tNM -- \t" << NM << "\tCOLLA -- \t\t" << TOTCOL << endl ;
        file_13 << "FTIME/TNORM , FLOAT(NM)/FLOAT(NMI)  -- " << FTIME/TNORM << (double)(NM)/(NMI) << endl  ;
        t[0] = clock() ;
         //cout<< "  TIME --   "<<setw(20)<<setprecision(10)<<FTIME<<"  NM  "<<NM<<" \t COLLS --  "<<std::left<<setw(20)<<setprecision(10)<<TOTCOL<<"\tCollision_time : "<<endl;
         t[1] = clock() ;
        MOLECULES_MOVE_1D() ;
       
      
        //cout  << "ITYPE[1] = " << ITYPE[1] << "\t ITYPE[2] = " << ITYPE[2] << endl ;
        t[2]= clock() ;
        if((ITYPE[1] == 0)||(ITYPE[2] == 0)||(ITYPE[2]==4))
            MOLECULES_ENTER_1D() ;
        t[3] = clock() ;

        INDEX_MOLS() ;
        t[4] = clock() ;
        
        COLLISIONS() ;
        t[5] = clock() ;

        

        if (MMVM>0) DISSOCIATION() ;
        t[6] = clock() ;

        if (FTIME > TSAMP)
        {
            if(ISF==0)  SAMPLE_FLOW() ;

            if((ISF == 1) && (FTIME < TPOUT+(1e00-FRACSAM)*DTOUT))
            {
                TSAMP =TSAMP+DTSAMP ;
                INITIALISE_SAMPLES() ;
            }
            if((ISF == 1) && (FTIME >= TPOUT+(1-FRACSAM)*DTOUT))
            {
                SAMPLE_FLOW() ;
            }
        }
        t[7] = clock() ;
        if(FTIME >TOUT )
        {
            OUTPUT_RESULTS() ;
            TPOUT = FTIME ;
        }
        t[8] = clock() ;

        for(int i= 0 ; i<8 ; i++)
        {
            cout << t[i+1]-t[i] << "\t" ;
        }
        cout << endl ;
        ch++ ;
        if(ch>10000)   
            break ;
    }

    cin >> TOTCOL ;
    return 0;
}


void ALLOCATE_GAS()
{
    // //GAS gas;
    // //CALC calc;
    d_allocate(MSP,2,FSP);

    d_allocate(6,MSP,SP);
    d_allocate(3,MSP,SPR);

    d_allocate(8,MSP,MSP,SPM);
    
    i_allocate(2,MSP,ISPR);
    i_allocate(MSP,ISPV);
    d_allocate(6,MSP,2,ENTR);
    d_allocate(MSP,2,VMP);
    d_allocate(MSP,VNMAX);
    d_allocate(MSP,CR);
    d_allocate(MSP,MSP,TCOL);
    i_allocate(MSP,MSP,ISPRC);
    i_allocate(MSP,MSP,ISPRK);
    d_allocate(4,MSP,MSP,MSP,SPRC);
    i_allocate(MSP,NELL);
    d_allocate(3,MELE+1,MSP,QELC);
    d_allocate(2,MSP,MSP,MVIBL+1,SPRP);
    d_allocate(2,MSP,MSP,SPRT);
    d_allocate(MSP,AJM);
    d_allocate(MSP,FP);
    d_allocate(MSP,ALOSS);
    d_allocate(MSP,EME);
    /*ALLOCATE (FSP(MSP,2),SP(6,MSP),SPR(3,MSP),SPM(8,MSP,MSP),ISPR(2,MSP),ISPV(MSP),ENTR(6,MSP,2),      &
     VMP(MSP,2),VNMAX(MSP),CR(MSP),TCOL(MSP,MSP),ISPRC(MSP,MSP),ISPRK(MSP,MSP),SPRC(4,MSP,MSP,MSP),                        &
     NELL(MSP),QELC(3,MELE,MSP),SPRP(2,MSP,MSP,0:MVIBL),SPRT(2,MSP,MSP),AJM(MSP),FP(MSP),    &
     ALOSS(MSP),EME(MSP),STAT=ERROR)
     //
     IF (ERROR /= 0) THEN
     WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SPECIES VARIABLES',ERROR
     END IF
     //*/
    i_allocate(MMEX,MSP,MSP,NEX);
    i_allocate(MSP,MSP,NSPEX);
    d_allocate(6,MMEX,MSP,MSP,SPEX);
    i_allocate(MMEX,7,MSP,MSP,ISPEX);
    i_allocate(4,MSP,TREACG);
    d_allocate(MMEX,PSF);
    i_allocate(4,MSP,TREACL);
    d_allocate(MEX,TNEX);
    d_allocate(2,MMEX,MSP,MSP,MVIBL+1,SPREX);
    i_allocate(2,MSP,NSLEV);
    d_allocate(MSP,SLER);
    // ALLOCATE (NEX(MMEX,MSP,MSP),NSPEX(MSP,MSP),SPEX(6,MMEX,MSP,MSP),ISPEX(MMEX,7,MSP,MSP),TREACG(4,MSP),         &
    //           PSF(MMEX),TREACL(4,MSP),TNEX(MEX),SPREX(2,MMEX,MSP,MSP,0:MVIBL),NSLEV(2,MSP),SLER(MSP),STAT=ERROR)
    // //
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE Q-K REACTION VARIABLES',ERROR
    // END IF
    // //
    

    if(MMVM >= 0){
        d_allocate(5,MMVM,MSP,SPVM);
        i_allocate(2,MMVM,MSP,ISPVM);
        d_allocate(MSP,TDISS);
        d_allocate(MSP,TRECOMB);
        //ALLOCATE (SPVM(5,MMVM,MSP),ISPVM(2,MMVM,MSP),TDISS(MSP),TRECOMB(MSP),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE VIBRATION VARIABLES',ERROR
    }
    
    //N.B. surface reactions are not yet implemented
    if(MNSR > 0){
        d_allocate(MNSR,ERS);
        i_allocate(2,MNSR,LIS);
        i_allocate(6,MNSR,LRS);
        i_allocate(MNSR,MSP,ISRCD);
        //ALLOCATE (ERS(MNSR),LIS(2,MNSR),LRS(6,MNSR),ISRCD(MNSR,MSP),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SURFACE REACTION VARIABLES',ERROR
    }
   

     //AJM=0.e00;
    //memget(AJM,0.e00,sizeof(*AJM));
    for(int i=0 ; i<MSP+1 ; i++ ) {
        AJM[i] = 0 ;
    }
    
    
    return;
    
}

void ENERGY(int I,double &TOTEN)
{
    //calculate the total energy (all molecules if I=0, otherwise molecule I)
    //I>0 used for dianostic purposes only
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,L,N,II,M,IV,KV,J;
    double TOTENI,TOTELE;
    //
    TOTEN=0.0;
    TOTELE=0;

    //
    
    if(I == 0){
        for(N=1;N<=NM;N++) {
            if( get(IPCELL  ,N) > 0){
                L=get(IPSP ,N) ;
                TOTENI=TOTEN;
                TOTEN=TOTEN+get(SP ,6,L);
                TOTEN=TOTEN+0.5e00*get(SP ,5,L)*(pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2)+pow(get(PV  ,3,N),2));
                if(get(ISPR ,1,L) > 0) TOTEN=TOTEN+PROT[N];
                if(get(ISPV  ,L) > 0){
                    for(KV=1;KV<=get(ISPV  ,L);KV++){
                        J=get (IPVIB , KV,N);
                        //         IF (J <0) THEN
                        //           J=-J
                        //           IF (J == 99999) J=0
                        //         END IF
                        TOTEN=TOTEN+double(J)*BOLTZ*get(SPVM ,1,KV,L);
                    }
                }
            }
            if(MELE > 1){
                TOTEN=TOTEN+PELE[N];
                TOTELE=TOTELE+PELE[N];
            }
            //if((TOTEN-TOTENI) > 1.e-16) cout<<"MOL "<<N<<" ENERGY "<<TOTEN-TOTENI<<endl;
        }
        //
        //WRITE (9,*) 'Total Energy =',TOTEN,NM
        //WRITE (*,*) 'Total Energy =',TOTEN,NM
        file_9<<"Total Energy =  "<<setprecision(25)<<TOTEN<<"\t"<<NM<<endl;
        cout<<"Total Energy =  "<<setprecision(20)<<TOTEN<<"\t"<<NM<<endl;
        //  WRITE (*,*) 'Electronic Energy =',TOTELE
    }
    else{
        N=I;
        if(get(IPCELL  ,N) > 0){
            L=get(IPSP ,N);
            TOTEN=TOTEN+get(SP ,6,L);
            TOTEN=TOTEN+0.5e00*get(SP ,5,L)*(pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2)+pow(get(PV  ,3,N),2));
            if(get(ISPR ,1,L) > 0) TOTEN=TOTEN+PROT[N];
            if(get(ISPV  ,L) > 0){
                for(KV=1;KV<=get(ISPV  ,L);KV++){
                    J=get (IPVIB , KV,N);
                    //         IF (J <0) THEN
                    //           J=-J
                    //           IF (J == 99999) J=0
                    //         END IF
                    TOTEN=TOTEN+double(J)*BOLTZ*get(SPVM ,1,KV,L);
                }
            }
        }
    }
    
    //
    return;   //
}

void INITIALISE_SAMPLES()
{
    file_9 << " INITIALISE SAMPLES IS RUNNING \n" ;
    file_3 << "INITIALISE SAMPLES IS running\n" ;
    cout << "INITIALISE SAMPLES IS running\n" ;
    int N;
    //
    NSAMP=0.0;
    TISAMP=FTIME;
    NMISAMP=NM;
    //memget(COLLS,0.e00,sizeof(*COLLS));memget(WCOLLS,0.e00,sizeof(*WCOLLS));memget(CLSEP,0.e00,sizeof(*CLSEP));
   
    for(int i=0;i<NCELLS+1;i++)
        COLLS[i]=0.e00;
    for(int i=0;i<NCELLS+1;i++)
       WCOLLS[i]=0.e00;
    for(int i=0;i<NCELLS+1;i++)
        CLSEP[i]=0.e00;
    //COLLS=0.e00 ; WCOLLS=0.e00 ; CLSEP=0.e00;
    //memget(TCOL,0.0,sizeof(*TCOL));//TCOL=0.0;
    for(int i=0;i<MSP+1;i++){
        for(int j=0;j<MSP+1;j++){
            get (TCOL , i,j)=0.0;
        }
    }
    //TREACG=0;
    //TREACL=0;
    for(int i=0;i<5;i++){
        for(int j=0;j<MSP+1;j++){
            get (TREACG , i,j)=0;
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<MSP+1;j++){
            get (TREACL , i,j)=0;
        }
    }
    //memget(CS,0.0,sizeof(*CS));memget(CSS,0.0,sizeof(*CSS));memget(CSSS,0.0,sizeof(*CSSS));
    for(int j=0;j<MSP+10;j++){
        for(int k=0;k<NCELLS+1;k++){
            for(int l=0;l<MSP+1;l++)
                get (CS,1+ j,k,l)=0.0;
        }
    }
    for(int i=0;i<9;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<MSP+1;k++){
                for(int l=0;l<3;l++)
                    get (CSS , 1+ i,j,k,l)=0.0;
            }
        }
    }
    for(int k=0;k<7;k++){
        for(int l=0;l<3;l++)
            get (CSSS , k,l)=0.0;
    }
    //CS=0.0 ; CSS=0.0 ; CSSS=0.0;
    //memget(VIBFRAC,0.e00,sizeof(*VIBFRAC));//VIBFRAC=0.e00;
    //memget(SUMVIB,0.e00,sizeof(*SUMVIB));//SUMVIB=0.e00;
    for(int j=0;j<MSP+1;j++){
        for(int k=0;k<MMVM+1;k++){
            for(int l=0;l<151;l++)
                get (VIBFRAC , j,k,l+1)=0.0;
        }
    }
    for(int k=0;k<MSP+1;k++){
        for(int l=0;l<MMVM+1;l++)
            get (SUMVIB , k,l)=0.0;
    }
    
}

void SET_INITIAL_STATE_1D()
{
    //set the initial state of a homogeneous or one-dimensional flow
    //
    //MOLECS molecs;
    //GEOM_1D geom;
    //GAS gas;
    //CALC calc;
    //OUTPUT output;
    //
    //
    int J,L,K,KK,KN,II,III,INC,NSET,NSC;
    long long N,M;
    double A,B,AA,BB,BBB,SN,XMIN,XMAX,WFMIN,DENG,ELTI,EA,XPREV;
    double DMOM[4];
    double VB[4][3];
    double ROTE[3];
    //
    //NSET the alternative set numbers in the setting of exact initial state
    //DMOM(N) N=1,2,3 for x,y and z momentum sums of initial molecules
    //DENG the energy sum of the initial molecules
    //VB alternative sets of velocity components
    //ROTE alternative sets of rotational energy
    //EA entry area
    //INC counting increment
    //ELTI  initial electronic temperature
    //XPREV the pevious x coordinate
    //
    //memget(DMOM,0.e00,sizeof(DMOM));
    for(int i=0;i<4;i++)
        DMOM[i]=0.e00;
    DENG=0.e00;
    //set the number of molecules, divisions etc. based on stream 1
    //
    NMI=10000*IMEG+2;    //small changes in number for statistically independent runs
    NDIV=NMI/MOLSC; //MOLSC molecules per division
    //WRITE (9,*) 'The number of divisions is',NDIV
    file_9<< "The number of divisions is "<<NDIV<<endl;
    //
    MDIV=NDIV;
    ILEVEL=0;
    //
    i_allocate(ILEVEL+1,MDIV,JDIV);
    // ALLOCATE (JDIV(0:ILEVEL,MDIV),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR JDIV ARRAY',ERROR
    // ENDIF
    //
    DDIV=(XB[2]-XB[1])/double(NDIV);
    NCELLS=NDIV;
    
    //WRITE (9,*) 'The number of sampling cells is',NCELLS
    file_9<<"The number of sampling cells is "<< NCELLS<<endl;
    NCIS=MOLSC/NMCC;
    NCCELLS=NCIS*NDIV;
    //WRITE (9,*) 'The number of collision cells is',NCCELLS
    file_9<< "The number of collision cells is "<<NCCELLS<<endl;
    //
    if(IFX == 0) XS=0.e00;
    //
    if(ISECS == 0){
        if(IFX == 0) FNUM=((XB[2]-XB[1])*FND[1])/double(NMI);
        if(IFX == 1) FNUM=PI*(pow(XB[2],2)-pow(XB[1],2))*FND[1]/double(NMI);
        if(IFX == 2) FNUM=1.3333333333333333333333e00*PI*(pow(XB[2],3)-pow(XB[1],3))*FND[1]/double(NMI);
    }
    else{
        if(IFX == 0) FNUM=((XS-XB[1])*FND[1]+(XB[2]-XS)*FND[2])/double(NMI);
        if(IFX == 1) FNUM=PI*((pow(XS,2)-pow(XB[1],2))*FND[1]+(pow(XB[2],2)-pow(XS,2))*FND[2])/double(NMI);
        if(IFX == 2) FNUM=1.3333333333333333333333e00*PI*((pow(XS,3)-pow(XB[1],3))*FND[1]+(pow(XB[2],3)-pow(XS,3))*FND[2])/double(NMI);
    }
    //
    FNUM=FNUM*FNUMF;
    if(FNUM < 1.e00) FNUM=1.e00;
    //
    FTIME=0.e00;
    //
    TOTMOV=0.e00;
    TOTCOL=0.e00;
    
    NDISSOC=0;
    //memget(TCOL,0.e00,sizeof(*TCOL));//TCOL=0.e00;
    for(int i=0;i<MSP+1;i++){
        for(int j=0;j<MSP+1;j++){
            get (TCOL , i,j)=0.e00;
        }
    }
    
    //memget(TDISS,0.e00,sizeof(*TDISS));//TDISS=0.e00;
    //memget(TRECOMB,0.e00,sizeof(*TRECOMB));//TRECOMB=0.e00;
    for(int i=0;i<MSP+1;i++)
        TDISS[i]=0.e00;
    for(int i=0;i<MSP+1;i++)
        TRECOMB[i]=0.e00;
    //TREACG=0;
    //TREACL=0;
    for(int i=0;i<5;i++){
        for(int j=0;j<MSP+1;j++){
            get (TREACG , i,j)=0;
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<MSP+1;j++){
            get (TREACL , i,j)=0;
        }
    }
    //memget(TNEX,0.e00,sizeof(*TNEX));//TNEX=0.e00;
    for(int i=0;i<MEX+1;i++)
        TNEX[i]= 0.e00;
    for(N=1;N<=NDIV;N++){
        get(JDIV ,1 , N)=-N;
    }
    
    //
    d_allocate(4,NCELLS,CELL);
    i_allocate(NCELLS,ICELL);
    d_allocate(5,NCCELLS,CCELL);
    i_allocate(3,NCCELLS,ICCELL);
    // ALLOCATE (CELL(4,NCELLS),ICELL(NCELLS),CCELL(5,NCCELLS),ICCELL(3,NCCELLS),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR CELL ARRAYS',ERROR
    // ENDIF
    //
    d_allocate(NCELLS,COLLS);
    d_allocate(NCELLS,WCOLLS);
    d_allocate(NCELLS,CLSEP);
    d_allocate(MNSR,SREAC);
    d_allocate(23,NCELLS,VAR);
    d_allocate(13,NCELLS,MSP,VARSP);
    d_allocate(36+MSP,2,VARS);
    d_allocate(10+MSP,NCELLS,MSP,CS);
    d_allocate(9,2,MSP,2,CSS);
    d_allocate(6,2,CSSS);
    
    // ALLOCATE (COLLS(NCELLS),WCOLLS(NCELLS),CLSEP(NCELLS),SREAC(MNSR),VAR(23,NCELLS),VARSP(0:12,NCELLS,MSP),    &
    //           VARS(0:35+MSP,2),CS(0:9+MSP,NCELLS,MSP),CSS(0:8,2,MSP,2),CSSS(6,2),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR SAMPLING ARRAYS',ERROR
    // ENDIF
    //
    if(MMVM >= 0){
        
        d_allocate(MSP,MMVM,151,VIBFRAC);
        d_allocate(MSP,MMVM,SUMVIB);
        // ALLOCATE (VIBFRAC(MSP,MMVM,0:150),SUMVIB(MSP,MMVM),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR RECOMBINATION ARRAYS',ERROR
        // ENDIF
    }
    //
    INITIALISE_SAMPLES();
    //
    //Set the initial cells
    
    for(N=1;N<=NCELLS;N++){
        get (CELL , 2,N)=XB[1]+double(N-1)*DDIV;
        get (CELL , 3,N)=get (CELL , 2,N)+DDIV;
        get (CELL , 1,N)=get (CELL , 2,N)+0.5e00*DDIV;
        if(IFX == 0) get (CELL , 4,N)=get (CELL , 3,N)-get (CELL , 2,N);    //calculation assumes unit cross-section
        if(IFX == 1) get (CELL , 4,N)=PI*(pow(get (CELL , 3,N),2)-pow(get (CELL , 2,N),2));  //assumes unit length of full cylinder
        if(IFX == 2) get (CELL , 4,N)=1.33333333333333333333e00*PI*(pow(get (CELL , 3,N),3)-pow(get (CELL , 2,N),3));    //flow is in the full sphere
        get (ICELL , N)=NCIS*(N-1);
        for(M=1;M<=NCIS;M++){
            L=get (ICELL , N)+M;
            XMIN=get (CELL , 2,N)+(M-1)*DDIV/double(NCIS);
            XMAX=XMIN+DDIV/double(NCIS);
            if(IFX == 0) get (CCELL , 1,L)=XMAX-XMIN;
            if(IFX == 1) get (CCELL , 1,L)=PI*(pow(XMAX,2)-pow(XMIN,2));  //assumes unit length of full cylinder
            if(IFX == 2) get (CCELL , 1,L)=1.33333333333333333333e00*PI*(pow(XMAX,3)-pow(XMIN,3));    //flow is in the full sphere
            get(CCELL , 2,L)=0.e00;
            get(ICCELL , 3,L)=N;
        }
        get(VAR , 11,N)=FTMP[1];
        get(VAR , 8,N)=FTMP[1];
    }
    //
    if(IWF == 0) AWF=1.e00;
    if(IWF == 1){
        //FNUM must be reduced to allow for the weighting factors
        A=0.e00;
        B=0.e00;
        for(N=1;N<=NCELLS;N++){
            A=A+get (CELL , 4,N);
            B=B+get (CELL , 4,N)/(1.0+WFM*pow(get (CELL , 1,N),IFX));
        }
        AWF=A/B;
        FNUM=FNUM*B/A;
    }
    //
    //WRITE (9,*) 'FNUM is',FNUM
    file_9<<"FNUM is "<<FNUM<<endl;
    //
    //set the information on the molecular species
    //
    A=0.e00;
    B=0.e00;
    for(L=1;L<=MSP;L++){
        A=A+get(SP ,5,L)*get(FSP ,L,1);
        B=B+(3.0+get(ISPR ,1,L))*get(FSP ,L,1);
        get (VMP , L,1)=sqrt(2.e00*BOLTZ*FTMP[1]/get(SP ,5,L));
        if((ITYPE[2]== 0) || (ISECS == 1)) get (VMP , L,2)=sqrt(2.e00*BOLTZ*FTMP[2]/get(SP ,5,L));
        VNMAX[L]=3.0*get (VMP , L,1);
        if(L == 1)
            VMPM=get (VMP , L,1);
        else
            if(get (VMP , L,1) > VMPM) VMPM=get (VMP , L,1);
    }
    //WRITE (9,*) 'VMPM =',VMPM
    file_9<< "VMPM = "<<VMPM<<endl;
    FDEN=A*FND[1];
    FPR=FND[1]*BOLTZ*FTMP[1];
    FMA=VFX[1]/sqrt((B/(B+2.e00))*BOLTZ*FTMP[1]/A);
    //set the molecular properties for collisions between unlike molecles
    //to the average of the molecules
    for(L=1;L<=MSP;L++){
        for(M=1;M<=MSP;M++){
            get (SPM , 4,L,M)=0.5e00*(get(SP ,1,L)+get(SP ,1,M));
            get (SPM , 3,L,M)=0.5e00*(get(SP ,3,L)+get(SP ,3,M));
            get (SPM , 5,L,M)=0.5e00*(get(SP ,2,L)+get(SP ,2,M));
            get (SPM , 1,L,M)=get(SP ,5,L)*(get(SP ,5,M)/(get(SP ,5,L)+get(SP ,5,M)));
            get (SPM , 2,L,M)=0.25e00*PI*pow((get(SP ,1,L)+get(SP ,1,M)),2);
            AA=2.5e00-get (SPM , 3,L,M);
            A=tgamma(AA);
            get (SPM , 6,L,M)=1.e00/A;
            get (SPM , 8,L,M)=0.5e00*(get(SP ,4,L)+get(SP ,4,M));
            if((get(ISPR ,1,L) > 0) && (get(ISPR ,1,M) > 0))
                get (SPM , 7,L,M)=(get(SPR ,1,L)+get(SPR ,1,M))*0.5e00;
            if((get(ISPR ,1,L) > 0) && (get(ISPR ,1,M) == 0))
                get (SPM , 7,L,M)=get(SPR ,1,L);
            if((get(ISPR ,1,M) > 0) && (get(ISPR ,1,L) == 0))
                get (SPM , 7,L,M)=get(SPR ,1,M);
        }
    }
    if(MSP == 1){   //set unscripted variables for the simple gas case
        RMAS=get (SPM , 1,1,1);
        CXSS=get (SPM , 2,1,1);
        RGFS=get (SPM , 6,1,1);
    }
    //
    for(L=1;L<=MSP;L++){
        CR[L]=0.e00;
        for(M=1;M<=MSP;M++){   //set the equilibrium collision rates
            CR[L]=CR[L]+2.e00*SPI*pow(get (SPM , 4,L,M),2)*FND[1]*get(FSP ,M,1)*pow((FTMP[1]/get (SPM , 5,L,M)),(1.0-get (SPM , 3,L,M)))*sqrt(2.0*BOLTZ*get (SPM , 5,L,M)/get (SPM , 1,L,M));
        }
    }
    A=0.e00;
    for(L=1;L<=MSP;L++)
        A=A+get(FSP ,L,1)*CR[L];
    CTM=1.e00/A;
    //WRITE (9,*) 'Collision time in the stream is',CTM
    file_9<< "Collision time in the stream is "<<CTM << endl ;
    //
    for(L=1;L<=MSP;L++){
        FP[L]=0.e00;
        for(M=1;M<=MSP;M++){
            FP[L]=FP[L]+PI*pow(get (SPM , 4,L,M),2)*FND[1]*get(FSP ,M,1)*pow((FTMP[1]/get (SPM , 5,L,M)),(1.0-get (SPM , 3,L,M)))*sqrt(1.e00+get(SP ,5,L)/get(SP ,5,M));
        }
        FP[L]=1.e00/FP[L];
    }
    FPM=0.e00;
    for(L=1;L<=MSP;L++)
        FPM=FPM+get(FSP ,L,1)*FP[L];
    //WRITE (9,*) 'Mean free path in the stream is',FPM
    file_9<<"Mean free path in the stream is "<<FPM<<endl;
    //
    TNORM=CTM;
    if(ICLASS == 1) TNORM= (XB[2]-XB[1])/VMPM;     //there may be alternative definitions
    //
    //set the initial time step
    DTM=CTM*CPDTM;
    //
    if(fabs(VFX[1]) > 1.e-6)
        A=(0.5e00*DDIV/VFX[1])*TPDTM;
    else
        A=0.5e00*DDIV/VMPM;
    
    if(IVB == 1){
        B=0.25e00*DDIV/(fabs(VELOB)+VMPM);
        if(B < A) A=B;
    }
    if(DTM > A) DTM=A;
    //
    DTM=0.1e00*DTM;   //OPTIONAL MANUAL ADJUSTMENT that is generally used with a fixed time step (e.g for making x-t diagram)
    //
    DTSAMP=SAMPRAT*DTM;
    DTOUT=OUTRAT*DTSAMP;
    TSAMP=DTSAMP;
    TOUT=DTOUT;
    ENTMASS=0.0;
    //
    //WRITE (9,*) 'The initial value of the overall time step is',DTM
    file_9<< "The initial value of the overall time step is "<<DTM<<endl;
    //
    //initialise cell quantities associated with collisions
    //
    for(N=1;N<=NCCELLS;N++){
        get (CCELL , 3,N)=DTM/2.e00;
        get (CCELL , 4,N)=2.e00*VMPM*get (SPM , 2,1,1);
        RANF=(double) (rand()%100000)/100001;
        // RANDOM_NUMBER(RANF)
        get (CCELL , 2,N)=RANF;
        get (CCELL , 5,N)=0.e00;
    }
    //
    //set the entry quantities
    //
    for(K=1;K<=2;K++){
        if((ITYPE[K] == 0) || ((K == 2) && (ITYPE[K] == 4))){
            if(IFX == 0) EA=1.e00;
            if(IFX == 1) EA=2.e00*PI*XB[K];
            if(IFX == 2) EA=4.e00*PI*pow(XB[K],2);
            for(L=1;L<=MSP;L++){
                if(K == 1) SN=VFX[1]/get (VMP , L,1);
                if(K == 2) SN=-VFX[2]/get (VMP , L,2);
                AA=SN;
                A=1.e00+erf(AA);
                BB=exp(-pow(SN,2));
                get (ENTR , 3,L,K)=SN;
                get (ENTR , 4,L,K)=SN+sqrt(pow(SN,2)+2.e00);
                get (ENTR , 5,L,K)=0.5e00*(1.e00+SN*(2.e00*SN-get (ENTR , 4,L,K)));
                get (ENTR , 6,L,K)=3.e00*get (VMP , L,K);
                B=BB+SPI*SN*A;
                get (ENTR , 1,L,K)=EA*FND[K]*get(FSP ,L,K)*get (VMP , L,K)*B/(FNUM*2.e00*SPI);
                get (ENTR , 2,L,K)=0.e00;
            }
        }
    }
    //
    //Set the uniform stream
    //
    MNM=1.1e00*NMI;
    //
    if(MMVM > 0){
        d_allocate(NCLASS,MNM,PX);
        d_allocate(MNM,PTIM);
        d_allocate(MNM,PROT);
        i_allocate(MNM,IPCELL);
        i_allocate(MNM,IPSP);
        i_allocate(MNM,ICREF);
        i_allocate(MNM,IPCP);
        d_allocate(3,MNM,PV);
        i_allocate(MMVM,MNM,IPVIB);
        d_allocate(MNM,PELE);
        // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),     &
        //      get(IPVIB  ,MMVM,MNM),PELE(MNM),STAT=ERROR)
    }
    
    else{
        if(MMRM > 0){
            d_allocate(NCLASS,MNM,PX);
            d_allocate(MNM,PTIM);
            d_allocate(MNM,PROT);
            i_allocate(MNM,IPCELL);
            i_allocate(MNM,IPSP);
            i_allocate(MNM,ICREF);
            i_allocate(MNM,IPCP);
            d_allocate(3,MNM,PV);
            d_allocate(MNM,PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
        else{
            d_allocate(NCLASS,MNM,PX);
            d_allocate(MNM,PTIM);
            i_allocate(MNM,IPCELL);
            i_allocate(MNM,IPSP);
            i_allocate(MNM,ICREF);
            i_allocate(MNM,IPCP);
            d_allocate(3,MNM,PV);
            d_allocate(MNM,PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR MOLECULE ARRAYS',ERROR
    // ENDIF
    //
    NM=0;
    if(IGS == 1){
        cout<<"Setting the initial gas"<<endl;
        for(L=1;L<=MSP;L++){
            //memget(ROTE,0.0,sizeof(ROTE));
            for(int i=0;i<3;i++)
                ROTE[i]=0.0;
            for(K=1;K<=ISECS+1;K++){
                if(ISECS == 0){         //no secondary stream
                    M=(double(NMI)*get(FSP ,L,1)*AWF);
                    XMIN=XB[1];
                    XMAX=XB[2];
                }
                else{
                    A=(pow(XS,JFX)-pow(XB[1],JFX))*FND[1]+(pow(XB[2],JFX)-pow(XS,JFX))*FND[2];
                    if(K == 1){
                        M=int(double(NMI)*((pow(XS,JFX)-pow(XB[1],JFX))*FND[1]/A)*get(FSP ,L,1));
                        XMIN=XB[1];
                        XMAX=XS;
                    }
                    else{
                        M=int(double(NMI)*((pow(XB[2],JFX)-pow(XS,JFX))*FND[2]/A)*get(FSP ,L,2));
                        XMIN=XS;
                        XMAX=XB[2];
                    }
                }
                if((K == 1) || (ISECS == 1)){
                    III=0;
                    WFMIN=1.e00+WFM*pow(XB[1],IFX);
                    N=1;
                    INC=1;
                    if((K== 2) && (JFX > 1)){
                        BBB=(pow(XMAX,JFX)-pow(XMIN,JFX))/double(M);
                        XPREV=XMIN;
                    }
                    while(N < M){
                        if((JFX == 1) || (K == 1))
                            A=pow((pow(XMIN,JFX)+(((N)-0.5e00)/(M))*pow((XMAX-XMIN),JFX)),(1.e00/double(JFX)));
                        else{
                            A=pow((pow(XPREV,JFX)+BBB),(1.e00/double(JFX)));
                            XPREV=A;
                        }
                        if(IWF == 0)
                            B=1.e00;
                        else{
                            B=WFMIN/(1.e00+WFM*pow(A,IFX));
                            if((B < 0.1e00) && (INC == 1)) INC=10;
                            if((B < 0.01e00) && (INC == 10)) INC=100;
                            if((B < 0.001e00) && (INC == 100)) INC=1000;
                            if((B < 0.0001e00) && (INC == 1000)) INC=10000;
                        }
                        RANF=((double)rand()/(double)RAND_MAX);
                        // CALL RANDOM_NUMBER(RANF)
                        if(B*double(INC) > RANF){
                            NM=NM+1;
                            get (PX , 1,NM)=A;
                            get(IPSP ,NM)=L;
                            PTIM[NM]=0.0;
                            if(IVB == 0) FIND_CELL_1D(get (PX , 1,NM),get(IPCELL  ,NM),KK);
                            if(IVB == 1) FIND_CELL_MB_1D(get (PX , 1,NM),get(IPCELL  ,NM),KK,PTIM[NM]);
                            //
                            for(NSET=1;NSET<=2;NSET++){
                                for(KK=1;KK<=3;KK++){
                                    RVELC(A,B,get (VMP , L,K));
                                    if(A < B){
                                        if(DMOM[KK] < 0.e00)
                                            BB=B;
                                        else
                                            BB=A;
                                    }           
                                    else{
                                        if(DMOM[KK] < 0.e00)
                                            BB=A;
                                        else
                                            BB=B;
                                    }
                                    VB[KK][NSET]=BB;
                                }
                                if(get(ISPR ,1,L) > 0) SROT(L,FTMP[K],ROTE[NSET]);
                            }
                            A=(0.5e00*get(SP ,5,L)*(pow(VB[1][1],2)+pow(VB[2][1],2)+pow(VB[3][1],2))+ROTE[1])/(0.5e00*BOLTZ*FTMP[K])-3.e00-double(get(ISPR ,1,L));
                            B=(0.5e00*get(SP ,5,L)*(pow(VB[1][2],2)+pow(VB[2][2],2)+pow(VB[3][2],2))+ROTE[2])/(0.5e00*BOLTZ*FTMP[K])-3.e00-double(get(ISPR ,1,L));
                            if(A < B){
                                if(DENG < 0.e00)
                                    KN=2;
                                else
                                    KN=1;
                            }
                            else{
                                if(DENG < 0.e00)
                                    KN=1;
                                else
                                    KN=2;
                            }
                            
                            for(KK=1;KK<=3;KK++){
                                get(PV  ,KK,NM)=VB[KK][KN];
                                DMOM[KK]=DMOM[KK]+VB[KK][KN];
                            }
                            get(PV  ,1,NM)=get(PV  ,1,NM)+VFX[K];
                            get(PV  ,2,NM)=get(PV  ,2,NM)+VFY[K];
                            if(get(ISPR ,1,L) > 0) PROT[NM]=ROTE[KN];
                            //           PROT(NM)=0.d00       //uncomment for zero initial rotational temperature (Figs. 6.1 and 6.2)
                            if(KN == 1) DENG=DENG+A;
                            if(KN == 2) DENG=DENG+B;
                            if(MMVM > 0){
                                if(get(ISPV  ,L) > 0){
                                    for(J=1;J<=get(ISPV  ,L);J++)
                                        SVIB(L,FVTMP[K],get (IPVIB , J,NM),J);
                                }
                                ELTI=FVTMP[K];
                                if(MELE > 1) SELE(L,ELTI,PELE[NM]);
                            }
                        }
                        N=N+INC;
                    }
                }
            }
        }
        //
        //WRITE (9,*) 'DMOM',DMOM
        //WRITE (9,*) 'DENG',DENG
        file_9<<"DMOM "<<DMOM[1] << "\t" << DMOM[2] << "\t" << DMOM[3] << endl;
        file_9<<"DENG "<< DENG <<endl;
    }
    //
    NMI=NM;
    //
    
    //SPECIAL CODING FOR INITIATION OF COMBUSION IN H2-02 MIXTURE (FORCED IGNITION CASES in section 6.7)
    //set the vibrational levels of A% random molecules to 5
    //  A=0.05D00
    //  M=0.01D00*A*NM
    //  DO N=1,M
    //    CALL RANDOM_NUMBER(RANF)
    //    K=INT(RANF*DFLOAT(NM))+1
    //    get(IPVIB  ,1,K)=5
    //  END DO
    //
    SAMPLE_FLOW();
    //OUTPUT_RESULTS();
    TOUT=TOUT-DTOUT;
    return;
}

void RVELC(double &U,double &V,double &VMP)
{
    //CALC calc;
    //generates two random velocity components U and V in an equilibrium
    //gas with most probable speed VMP
    //based on equations (4.4) and (4.5)
    double A,B;
    //
    // CALL RANDOM_NUMBER(RANF)
    RANF=((double)rand()/(double)RAND_MAX) ;
    A=sqrt(-log(RANF)) ;
    // CALL RANDOM_NUMBER(RANF)
    RANF=((double)rand()/(double)RAND_MAX) ;
    B=DPI*RANF ;
    U=A*sin(B)*VMP ;
    V=A*cos(B)*VMP ;
    return ;
}

void SROT(int &L,double &TEMP,double &ROTE)
{
    //sets a typical rotational energy ROTE of species L
    //CALC calc;
    //GAS gas;
    //
    // IMPLICIT NONE
    //
    int I;
    double A,B,ERM;
    //
    if(get(ISPR ,1,L) == 2){
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        ROTE=-log(RANF)*BOLTZ*TEMP;   //equation (4.8)
    }
    else{
        A=0.5e00*get(ISPR ,1,L)-1.e00;
        I=0;
        while(I == 0){
            // CALL RANDOM_NUMBER(RANF)
            RANF=((double)rand()/(double)RAND_MAX);
            ERM=RANF*10.e00;
            //there is an energy cut-off at 10 kT
            B=(pow((ERM/A),A))*exp(A-ERM);      //equation (4.9)
            // CALL RANDOM_NUMBER(RANF)
            RANF=((double)rand()/(double)RAND_MAX);
            if(B > RANF) I=1;
        }
        ROTE=ERM*BOLTZ*TEMP;
    }
    return;
}

void SVIB(int &L,double &TEMP,int &IVIB, int &K)
{
    //sets a typical vibrational state at temp. TEMP of mode K of species L
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int N;
    //    double TEMP;
    //    int IVIB;
    //
    // CALL RANDOM_NUMBER(RANF)
    RANF=((double)rand()/(double)RAND_MAX);
    N=-log(RANF)*TEMP/get(SPVM ,1,K,L);                 //eqn(4.10)
    //the state is truncated to an integer
    IVIB=N;
}

void SELE(int &L,double &TEMP, double &ELE)
{
    //sets a typical electronic energy at temp. TEMP of species L
    //employs direct sampling from the Boltzmann distribution
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,N;
    double EPF,A,B;
    double CTP[20];
    //
    //ELE electronic energy of a molecule
    //EPF electronic partition function
    //CTP(N) contribution of electronic level N to the electronic partition function
    //
    if(TEMP > 0.1){
        EPF=0.e00;
        for(N=1;N<=get(NELL  ,L);N++)
            EPF=EPF+get(QELC ,1,N,L)*exp(-get(QELC ,2,N,L)/TEMP) ;
        //
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        //
        A=0.0;
        K=0; //becomes 1 when the energy is set
        N=0;  //level
        while(K == 0){
            N=N+1;
            A=A+get(QELC ,1,N,L)*exp(-get(QELC ,2,N,L)/TEMP);
            B=A/EPF;
            if(RANF < B){
                K=1;
                ELE=BOLTZ*get(QELC ,2,N,L);
            }
        }
    }
    else
        ELE=0.e00;
    
    //
}

void CQAX(double &A,double &X,double &GAX)
{
    //calculates the function Q(a,x)=Gamma(a,x)/Gamma(a)
    //
    // IMPLICIT NONE
    double G,DT,T,PV,V;
    int NSTEP,N;
    //
    G=tgamma(A);
    //
    if(X < 10.e00){       //direct integration
        NSTEP=100000;
        DT=X/double(NSTEP);
        GAX=0.e00;
        PV=0.e00;
        for(N=1;N<=NSTEP;N++){
            T=double(N)*DT;
            V=exp(-T)*pow(T,(A-1));
            GAX=GAX+(PV+V)*DT/2.e00;
            PV=V;
        }
        GAX=1.e00-GAX/G;
    }
    else{      //asymptotic formula
        GAX=pow(X,(A-1.e00))*exp(-X)*(1.0+(A-1.e00)/X+(A-1.e00)*(A-2.e00)/pow(X,2)+(A-1.e00)*(A-2.e00)*(A-3.e00)/pow(X,3)+(A-1.e00)*(A-2.e00)*(A-3.e00)*(A-4.e00)/pow(X,4));
        GAX=GAX/G;
    }
    //
    return;
}
//****
//
void LBS(double XMA,double XMB,double &ERM)
{
    //selects a Larsen-Borgnakke energy ratio using eqn (11.9)
    //
    double PROB,RANF;
    int I,N;
    //
    //I is an indicator
    //PROB is a probability
    //ERM ratio of rotational to collision energy
    //XMA degrees of freedom under selection-1
    //XMB remaining degrees of freedom-1
    //
    I=0;
    while(I == 0){
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        ERM=RANF;
        if((XMA < 1.e-6) || (XMB < 1.e-6)){
            //    IF (XMA < 1.E-6.AND.XMB < 1.E-6) RETURN
            //above can never occur if one mode is translational
            if(XMA < 1.e-6) PROB=pow((1.e00-ERM),XMB);
            if(XMB < 1.e-6) PROB=pow((1.e00-ERM),XMA);
        }
        else
            PROB=pow(((XMA+XMB)*ERM/XMA),XMA)*pow(((XMA+XMB)*(1.e00-ERM)/XMB),XMB);
        
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        if(PROB > RANF) I=1;
    }
    //
    return;
}

void FIND_CELL_1D(double &X,int &NCC,int &NSC)
{
    //find the collision and sampling cells at a givem location in a 0D or 1D case
    //MOLECS molecs;
    //GEOM_1D geom;
    //CALC calc;
    
    int N,L,M,ND;
    double FRAC,DSC;
    //
    //NCC collision cell number
    //NSC sampling cell number
    //X location
    //ND division number
    //DSC the ratio of the sub-division width to the division width
    //
    ND=(X-XB[1])/DDIV+0.99999999999999e00 ;
    //
    if(get(JDIV ,1,ND) < 0){    //the division is a level 0 (no sub-division) sampling cell
        NSC=-get(JDIV ,1,ND);
        //  IF (IFX == 0)
        NCC=NCIS*(X-get (CELL , 2,NSC))/(get (CELL , 3,NSC)-get (CELL , 2,NSC))+0.9999999999999999e00;
        NCC=NCC+get (ICELL , NSC);
        //  IF (NCC == 0) NCC=1
        return;
    }
    else{  //the molecule is in a subdivided division
        FRAC=(X-XB[1])/DDIV-double(ND-1);
        M=ND;
        for(N=1;N<=ILEVEL;N++){
            DSC=1.e00/double(N+1);
            for(L=1;L<=2;L++){  //over the two level 1 subdivisions
                if(((L == 1) && (FRAC < DSC)) || ((L == 2) || (FRAC >= DSC))){
                    M=get(JDIV ,N,M)+L;  //the address in JDIV
                    if(get(JDIV ,N+1,M) < 0){
                        NSC=-get(JDIV ,N+1,M);
                        NCC=NCIS*(X-get (CELL , 2,NSC))/(get (CELL , 3,NSC)-get (CELL , 2,NSC))+0.999999999999999e00;
                        if(NCC == 0) NCC=1;
                        NCC=NCC+get (ICELL , NSC);
                        return;
                    }
                }
            }
            FRAC=FRAC-DSC;
        }
    }
   // file_9<<"No cell for molecule at x= "<<X<<endl; // dsuedit
    return ;
}

void FIND_CELL_MB_1D(double &X,int &NCC,int &NSC,double &TIM)
{
    //find the collision and sampling cells at a givem location in a 0D or 1D case
    //when there is a moving boundary
    //MOLECS molecs;
    //GEOM_1D geom;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int N,L,M,ND;
    double FRAC,DSC,A,B,C;
    //
    //NCC collision cell number
    //NSC sampling cell number
    //X location
    //ND division number
    //DSC the ratio of the sub-division width to the division width
    //TIM the time
    //
    A=(XB[2]+VELOB*TIM-XB[1])/double(NDIV);      //new DDIV
    ND=(X-XB[1])/A+0.99999999999999e00;
    B=XB[1]+double(ND-1)*A;
    //
    //the division is a level 0 sampling cell
    NSC=-get(JDIV ,1,ND);
    NCC=NCIS*(X-B)/A+0.99999999999999e00;
    NCC=NCC+get (ICELL , NSC);
    
    //WRITE (9,*) 'No cell for molecule at x=',X
    file_9<< "No cell for molecule at x= "<<X<<endl;
    return;
    //return ;
    //
}

void REFLECT_1D(int &N,int J,double &X)
{
    //reflects molecule N and samples the surface J properties
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int L,K,M;
    double A,B,VMPS,DTR,XI,DX,DY,DZ,WF;
    //
    //VMPS most probable velocity at the surface temperature
    //DTR time remaining after molecule hits a surface
    //
    L=get(IPSP ,N);
    WF=1.e00;
    if(IWF == 1) WF=1.e00+WFM*pow(X,IFX);
    get (CSS , 1+ 0,J,L,1)=get (CSS , 1+ 0,J,L,1)+1.e00;
    get (CSS , 1+ 1,J,L,1)=get (CSS , 1+ 1,J,L,1)+WF;
    get (CSS , 1+ 2,J,L,1)=get (CSS , 1+ 2,J,L,1)+WF*get(PV  ,1,N)*get(SP ,5,L);
    get (CSS , 1+ 3,J,L,1)=get (CSS , 1+ 3,J,L,1)+WF*(get(PV  ,2,N)-VSURF[J])*get(SP ,5,L);
    get (CSS , 1+ 4,J,L,1)=get (CSS , 1+ 4,J,L,1)+WF*get(PV  ,3,N)*get(SP ,5,L);
    A=pow(get(PV  ,1,N),2)+pow((get(PV  ,2,N)-VSURF[J]),2)+pow(get(PV  ,3,N),2);
    get (CSS , 1+ 5,J,L,1)=get (CSS , 1+ 5,J,L,1)+WF*0.5e00*get(SP ,5,L)*A;
    if(get(ISPR ,1,L) > 0) get (CSS , 1+ 6,J,L,1)=get (CSS , 1+ 6,J,L,1)+WF*PROT[N];
    if(MELE > 1) get (CSS , 1+ 8,J,L,1)=get (CSS , 1+ 8,J,L,1)+WF*PELE[N];
    if(MMVM > 0){
        if(get(ISPV  ,L) > 0){
            for(K=1;K<=get(ISPV  ,L);K++)
                get (CSS , 1+ 7,J,L,1)=get (CSS , 1+ 7,J,L,1)+WF*double(get (IPVIB , K,N))*BOLTZ*get(SPVM ,1,K,L);
        }
    }
    A=pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2)+pow(get(PV  ,3,N),2);
    B=fabs(get(PV  ,1,N));
    get (CSSS , 1,J)=get (CSSS , 1,J)+WF/B;
    get (CSSS , 2,J)=get (CSSS , 2,J)+WF*get(SP ,5,L)/B;
    get (CSSS , 3,J)=get (CSSS , 3,J)+WF*get(SP ,5,L)*get(PV  ,2,N)/B;
    //this assumes that any flow normal to the x direction is in the y direction
    get (CSSS , 4,J)=get (CSSS , 4,J)+WF*get(SP ,5,L)*A/B;
    if(get(ISPR ,1,L) > 0){
        get (CSSS , 5,J)=get (CSSS , 5,J)+WF*PROT[N]/B;
        get (CSSS , 6,J)=get (CSSS , 6,J)+WF*get(ISPR ,1,L)/B;
    }
    //
    // CALL RANDOM_NUMBER(RANF)
    RANF=((double)rand()/(double)RAND_MAX);
    if(FSPEC[J] > RANF){      //specular reflection
        X=2.e00*XB[J]-X;
        get(PV  ,1,N)=-get(PV  ,1,N);
        DTR=(X-XB[J])/get(PV  ,1,N);
    }
    else{                         //diffuse reflection
        VMPS=sqrt(2.e00*BOLTZ*TSURF[J]/get(SP ,5,L));
        DTR=(XB[J]-get (PX , 1,N))/get(PV  ,1,N);
        // CALL RANDOM_NUMBER(RANF)
        
        RANF=((double)rand()/(double)RAND_MAX);
        get(PV  ,1,N)=sqrt(-log(RANF))*VMPS;
        
        if(J == 2) get(PV  ,1,N)=-get(PV  ,1,N);
        RVELC(get(PV  ,2,N),get(PV  ,3,N),VMPS);
        get(PV  ,2,N)=get(PV  ,2,N)+VSURF[J];
        if(get(ISPR ,1,L) > 0) SROT(L,TSURF[J],PROT[N]);
        if(MMVM > 0){
            for(K=1;K<=get(ISPV  ,L);K++)
                SVIB(L,TSURF[J],get (IPVIB , K,N),K);
        }
        if(MELE > 1) SELE(L,TSURF[J],PELE[N]);
    }
    //
    get (CSS , 1+ 2,J,L,2)=get (CSS , 1+ 2,J,L,2)-WF*get(PV  ,1,N)*get(SP ,5,L);
    get (CSS , 1+ 3,J,L,2)=get (CSS , 1+ 3,J,L,2)-WF*(get(PV  ,2,N)-VSURF[J])*get(SP ,5,L);
    get (CSS , 1+ 4,J,L,2)=get (CSS , 1+ 4,J,L,2)-WF*get(PV  ,3,N)*get(SP ,5,L);
    A=pow(get(PV  ,1,N),2)+pow((get(PV  ,2,N)-VSURF[J]),2)+pow(get(PV  ,3,N),2);
    get (CSS , 1+ 5,J,L,2)=get (CSS , 1+ 5,J,L,2)-WF*0.5e00*get(SP ,5,L)*A;
    if(get(ISPR ,1,L) > 0) get (CSS , 1+ 6,J,L,2)=get (CSS , 1+ 6,J,L,2)-WF*PROT[N];
    if(MELE > 1) get (CSS , 1+ 8,J,L,2)=get (CSS , 1+ 8,J,L,2)-WF*PELE[N];
    if(MMVM > 0){
        if(get(ISPV  ,L) > 0){
            for(K=1;K<=get(ISPV  ,L);K++)
                get (CSS , 1+ 7,J,L,2)=get (CSS , 1+ 7,J,L,2)-WF*double(get (IPVIB , K,N))*BOLTZ*get(SPVM ,1,K,L);
        }
    }
    A=pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2)+pow(get(PV  ,3,N),2);
    B=fabs(get(PV  ,1,N));
    get (CSSS , 1,J)=get (CSSS , 1,J)+WF/B;
    get (CSSS , 2,J)=get (CSSS , 2,J)+WF*get(SP ,5,L)/B;
    get (CSSS , 3,J)=get (CSSS , 3,J)+WF*get(SP ,5,L)*get(PV  ,2,N)/B;
    //this assumes that any flow normal to the x direction is in the y direction
    get (CSSS , 4,J)=get (CSSS , 4,J)+WF*get(SP ,5,L)*A/B;
    if(get(ISPR ,1,L) > 0){
        get (CSSS , 5,J)=WF*get (CSSS , 5,J)+PROT[N]/B;
        get (CSSS , 6,J)=get (CSSS , 6,J)+WF*get(ISPR ,1,L)/B;
    }
    //
    XI=XB[J];
    DX=DTR*get(PV  ,1,N);
    DZ=0.e00;
    if(IFX > 0) DY=DTR*get(PV  ,2,N);
    if(IFX == 2) DZ=DTR*get(PV  ,3,N);
    if(IFX == 0) X=XI+DX;
    if(IFX > 0) AIFX(XI,DX,DY,DZ,X,get(PV  ,1,N),get(PV  ,2,N),get(PV  ,3,N));
    //
    return;
}

void DERIVED_GAS_DATA()
{
    //
    //GAS gas;
    //CALC calc;
    int I,II,J,JJ,K,L,M,MM,N,JMAX,MOLSP,MOLOF,NSTEP,IMAX;
    double A,B,BB,C,X,T,CUR,EAD,TVD,ZVT,ERD,PETD,DETD,PINT,ETD,SUMD,VAL;
    double *BFRAC,*TOT;
    double *VRRD;
    double *VRREX;
    //
    //VRRD(1,L,M,K) dissociation rate coefficient to species L,M for vibrational level K at 5,000 K
    //VRRD(2,L,M,K) similar for 15,000 K
    //VRREX(1,J,L,M,K)  Jth exchange rate coefficient to species L,M for vibrational level K at 1,000 K
    //VRREX(2,J,L,M,K) similar for 3,000 K
    //BFRAC(2,J) Boltzmann fraction
    //JMAX imax-1
    //T temperature
    //CUR sum of level resolved rates
    //
    
    d_allocate(2,MSP,MSP, MVIBL+1 , VRRD ) ;
    d_allocate(MVIBL+1,2 ,BFRAC ) ;
    d_allocate(2 ,MMEX , MSP,MSP, MVIBL+1 , VRREX ) ;
    d_allocate(MVIBL+1 , 2 , TOT ) ;
    
    // ALLOCATE (VRRD(2,MSP,MSP,0:MVIBL),BFRAC(0:MVIBL,2),VRREX(2,MMEX,MSP,MSP,0:MVIBL),TOT(0:MVIBL,2),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE VIB. RES. DISS. RATES',ERROR
    // END IF
    //
    cout<<"Setting derived gas data"<<endl;
    //copy the L,M data that has been specified for L < M so that it applies also for M>L
    for(L=1;L<=MSP;L++){
        for(M=1;M<=MSP;M++){
            if(L > M){
                get(NSPEX  ,L,M)=get(NSPEX  ,M,L);
                get(ISPRC ,L,M)=get(ISPRC ,M,L);
                get(ISPRK ,L,M)=get(ISPRK ,M,L);
                for(K=1;K<=MSP;K++){
                    get( SPRT ,1,L,M)=get( SPRT ,1,M,L);
                    get( SPRT ,2,L,M)=get( SPRT ,2,M,L);
                    get( SPRC ,1,L,M,K)=get( SPRC ,1,M,L,K);
                    get( SPRC ,2,L,M,K)=get( SPRC ,2,M,L,K);
                }
                for(K=1;K<=MMEX;K++){
                    get(NEX  ,K,L,M)=get(NEX  ,K,M,L);
                    for(J=1;J<=6;J++){
                        get(SPEX  ,J,K,L,M)=get(SPEX  ,J,K,M,L);
                    }
                    for(J=1;J<=7;J++){
                        get(ISPEX  ,K,J,L,M)=get(ISPEX  ,K,J,M,L);
                    }
                }
            }
        }
    }
    //
    if(MMVM > 0){
        //set the characteristic dissociation temperatures
        for(L=1;L<=MSP;L++){
            if(get(ISPV  ,L) > 0){
                for(K=1;K<=get(ISPV  ,L);K++)
                {
                    I=get( ISPVM ,1,K,L);
                    J=get( ISPVM ,2,K,L);
                    get(SPVM ,4,K,L)=(get(SP ,6,I)+get(SP ,6,J)-get(SP ,6,L))/BOLTZ;
                    //WRITE (9,*) 'Char. Diss temp of species',L,' is',SPVM(4,K,L)
                    file_9<<"Char. Diss temp of species "<<L<<" is "<<get(SPVM ,4,K,L)<<endl;
                }
            }
        }
    }
    //
    if(MMEX > 0){
        //set the heats of reaction of the exchange and chain reactions
        for(L=1;L<=MSP;L++){
            for(M=1;M<=MSP;M++){
                for(J=1;J<=MMEX;J++){
                    if((get(ISPEX  ,J,3,L,M)> 0) && (get(ISPEX  ,J,4,L,M)>0) && (get(ISPEX  ,J,1,L,M)>0) && (get(ISPEX  ,J,2,L,M)>0)){
                        get(SPEX  ,3,J,L,M)=get(SP ,6,get(ISPEX  ,J,1,L,M))+get(SP ,6,get(ISPEX  ,J,2,L,M))-get(SP ,6,get(ISPEX  ,J,3,L,M))-get(SP ,6,get(ISPEX  ,J,4,L,M));
                        // WRITE (9,*) 'Reaction',NEX(J,L,M),' heat of reaction',SPEX(3,J,L,M)
                        file_9<<"Reaction "<<get(NEX  ,J,L,M)<<" heat of reaction"<<get(SPEX  ,3,J,L,M)<<endl;
                    }
                }
            }
        }
    }
    //
    if(MELE > 1){
        //set the electronic cross-section ratios to a mean electronic relaxation collision number
        //(equipartition is not achieved unless there is a single number)
        for(L=1;L<=MSP;L++){
            A=0.e00;
            for(K=1;K<=get(NELL  ,L);K++){
                A=A+get(QELC ,3,K,L);
            }
            get(QELC ,3,1,L)=A/double(get(NELL  ,L));
        }
    }
    //
    //set the cumulative distributions of the post-recombination vibrational distributions for establishment of detailed balance
    for(L=1;L<=MSP;L++){
        for(M=1;M<=MSP;M++){
            if(get(ISPRC ,L,M) > 0){
                N=get(ISPRC ,L,M);   //recombined species
                K=get(ISPRK ,L,M);   //relevant vibrational mode
                //WRITE (9,*) 'SPECIES',L,M,' RECOMBINE TO',N
                file_9<<"SPECIES "<<L<<" "<<M<<" RECOMBINE TO"<<N<<endl;
                JMAX=get(SPVM ,4,K,N)/get(SPVM ,1,K,N);
                if(JMAX > MVIBL){
                    cout<<" The variable MVIBL="<<MVIBL<<" in the gas database must be increased to"<<JMAX<<endl;
                    cout<<"Enter 0 ENTER to stop";
                    cin>> A;
                    return ;
                }
                A=2.5e00-get(SP ,3,N);
                for(I=1;I<=2;I++){
                    if(I == 1) T=get( SPRT ,1,L,M);
                    if(I == 2) T=get( SPRT ,2,L,M);
                    //WRITE (9,*) 'TEMPERATURE',T
                    file_9<<"TEMPERATURE "<<T<<endl;
                    CUR=0.e00;
                    for(J=0;J<=JMAX;J++){
                        X=double(JMAX+1-J)*get(SPVM ,1,K,N)/T;
                        CQAX(A,X,B);
                        get (VRRD , I,L,M,J+1)=B*exp(-double(J)*get(SPVM ,1,K,N)/T);
                        CUR=CUR+get (VRRD , I,L,M,J+1);
                    }
                    B=0.e00;
                    for(J=0;J<=JMAX;J++){
                        B=B+get (VRRD , I,L,M,J+1)/CUR;
                        get (SPRP , I,L,M,J+1)=B;
                        //WRITE (9,*) 'CDF level dissoc',J,SPRP(I,L,M,J)
                        file_9<< "CDF level dissoc "<<J<<" "<<get (SPRP , I,L,M,J+1);
                    }
                }
            }
        }
    }
    //
    //READ (*,*)  //optionally pause program to check cumulative distributions for exchange and chain reactions
    //
    //set the cumulative distributions of the post-reverse vibrational distributions for establishment of detailed balance
    for(L=1;L<=MSP;L++){
        for(M=1;M<=MSP;M++){
            if(get(NSPEX  ,L,M) > 0){
                for(K=1;K<=get(NSPEX  ,L,M);K++){
                    if(get(SPEX  ,3,K,L,M) > 0.e00){         //exothermic (reverse) exchange reaction
                        //L,M are the species in the reverse reaction, E_a of forward reaction is SPEX(3,K,L,M)
                        //WRITE (9,*) 'SPECIES',L,M,' REVERSE REACTION'
                        file_9<<"SPECIES "<<L<<" "<<M<<" REVERSE REACTION"<<endl;
                        MOLSP=get(ISPEX  ,K,3,L,M);  //molecuke that splits in the forward reaction
                        MOLOF=get(ISPEX  ,K,4,L,M);
                        JMAX=(get(SPEX  ,3,K,L,M)+get(SPEX  ,6,K,MOLSP,MOLOF))/(BOLTZ*get(SPVM ,1,get(ISPEX  ,K,5,L,M),MOLSP))+15;   //should always be less than the JMAX set by dissociation reactions
                        for(I=1;I<=2;I++){
                            if(I == 1) T=get(SPEX  ,4,K,L,M);
                            if(I == 2) T=get(SPEX  ,5,K,L,M);
                            for(J=0;J<=JMAX;J++){
                                EAD=(get(SPEX  ,3,K,L,M)+get(SPEX  ,6,K,MOLSP,MOLOF))/(BOLTZ*T);
                                TVD=get(SPVM ,1,get(ISPEX  ,K,5,L,M),MOLSP)/T;
                                ZVT=1.e00/(1.e00-exp(-TVD));
                                C=ZVT/(tgamma(2.5e00-get(SP ,3,MOLSP))*exp(-EAD));  //coefficient of integral
                                ERD=EAD-double(J)*TVD;
                                if(ERD < 0.e00) ERD=0.e00;
                                PETD=ERD;
                                DETD=0.01e00;
                                PINT=0.e00;  //progressive value of integral
                                NSTEP=0;
                                A=1.e00;
                                while(A > 1.e-10){
                                    NSTEP=NSTEP+1;
                                    ETD=PETD+0.5e00*DETD;
                                    SUMD=0.e00;  //normalizing sum in the denominator
                                    IMAX=ETD/TVD+J;
                                    for(II=0;II<=IMAX;II++){
                                        SUMD=SUMD+pow((1.e00-double(II)*TVD/(ETD+double(J)*TVD)),(1.5e00-get(SP ,3,MOLSP)));
                                    }
                                    VAL=(pow((ETD*(1.e00-EAD/(ETD+double(J)*TVD))),(1.5e00-get(SP ,3,MOLSP)))/SUMD)*exp(-ETD);
                                    PINT=PINT+VAL*DETD;
                                    A=VAL/PINT;
                                    PETD=ETD+0.5e00*DETD;
                                }
                                get (VRREX , I,K,L,M,J+1)=C*PINT;
                                //              WRITE (*,*) 'Level ratio exch',I,J,VRREX(I,K,L,M,J)
                            }
                        }
                        //
                        //memget(TOT,0.e00,sizeof(*TOT));//TOT=0.e00;
                        for(int i=0;i<MVIBL+1;i++){
                            for(int j=0;j<MVIBL+1;j++){
                                get (TOT,1+ i,j)=0;
                            }
                        }
                        for(I=1;I<=2;I++){
                            if(I == 1) T=get(SPEX  ,4,K,L,M);
                            if(I == 2) T=get(SPEX  ,5,K,L,M);
                            for(J=0;J<=JMAX;J++){
                                TVD=get(SPVM ,1,get(ISPEX  ,K,5,L,M),MOLSP)/T;
                                ZVT=1.e00/(1.e00-exp(-TVD));
                                get (BFRAC , J+1,I)=exp(-J*get(SPVM ,1,get(ISPEX  ,K,5,L,M),MOLSP)/T)/ZVT;    //Boltzmann fraction
                                get (VRREX , I,K,L,M,J+1)=get (VRREX , I,K,L,M,J+1)*get (BFRAC , 1+J,I);
                                //              WRITE (*,*) 'Contribution',I,J,VRREX(I,K,L,M,J)
                                for(MM=0;MM<=J;MM++)
                                    get (TOT,1+ J,I)=get (TOT,1+ J,I)+get (VRREX , I,K,L,M,MM+1);
                            }
                        }
                        //
                        for(I=1;I<=2;I++){
                            for(J=0;J<=JMAX;J++){
                                get (SPREX , I,K,L,M,J+1 )=get (TOT,1+ J,I);
                                if(J == JMAX) get (SPREX , I,K,L,M,J+1)=1.e00;
                                //WRITE (9,*) 'Cumulative',I,J,SPREX(I,K,L,M,J)
                                file_9<<"Cumulative "<<I<<" "<<J<<" "<<get (SPREX , I,K,L,M,J+1);
                            }
                        }
                    }
                }
                NSLEV=0;
                //memget(SLER,0.e00,sizeof(*SLER));//SLER=0.e00;
                for(int i=0;i<MSP+1;i++)
                    SLER[i]=0.e00;
            }
        }
    }
    //
    //READ (*,*)  //optionally pause program to check cumulative distributions for exchange abd chain reactions
    return;
}


void MOLECULES_MOVE_1D()
{//
    //molecule moves appropriate to the time step
    //for homogeneous and one-dimensional flows
    //(homogeneous flows are calculated as one-dimensional)
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int N,L,M,K,NCI,J,II,JJ;
    double A,B,X,XI,XC,DX,DY,DZ,DTIM,S1,XM,R,TI,DTC,POB,UR,WFI,WFR,WFRI;
    //
    //N working integer
    //NCI initial cell time
    //DTIM time interval for the move
    //POB position of the outer boundary
    //TI initial time
    //DTC time interval to collision with surface
    //UR radial velocity component
    //WFI initial weighting factor
    //WFR weighting factor radius
    //WFRI initial weighting factor radius
    //

    if((ITYPE[2] == 4) && (ICN == 1)){
        //memget(ALOSS,0.e00,sizeof(*ALOSS));//ALOSS=0.e00;
        for(int i=0;i<MSP+1;i++)
            ALOSS[i]=0.e00;
        
        NMP=NM;
    }
    //
    N=1;
   
    while(N <= NM){
        //
        NCI=get(IPCELL  ,N);
        if((IMTS == 0) || (IMTS == 2)) DTIM=DTM;
        if(IMTS == 1) DTIM=2.e00*get (CCELL , 3,NCI);
        if(FTIME-PTIM[N] > 0.5*DTIM){
            WFI=1.e00;
            if(IWF == 1) WFI=1.e00+WFM*pow(get (PX , 1,N),IFX);
            II=0; //becomes 1 if a molecule is removed
            TI=PTIM[N];
            PTIM[N]=TI+DTIM;
            TOTMOV=TOTMOV+1;
            //
            XI=get (PX , 1,N);
            DX=DTIM*get(PV  ,1,N);
            X=XI+DX;
            //
            if(IFX > 0){
                DY=0.e00;
                DZ=DTIM*get(PV  ,3,N);
                if(IFX == 2) DY=DTIM*get(PV  ,2,N);
                R=sqrt(X*X+DY*DY+DZ*DZ);
            }
            //
            if(IFX == 0){
                for(J=1;J<=2;J++){    // 1 for minimum x boundary, 2 for maximum x boundary
                    if(II == 0){
                        if(((J == 1) && (X < XB[1])) || ((J == 2) && (X > (XB[2]+VELOB*PTIM[N])))){  //molecule crosses a boundary
                            if((ITYPE[J] == 0) || (ITYPE[J] == 3) || (ITYPE[J] == 4)){
                                if(XREM > XB[1]){
                                    L=get(IPSP ,N);
                                    ENTMASS=ENTMASS-get(SP ,5,L);
                                }
                                if((ITYPE[2] == 4) && (ICN == 1)){
                                    L=get(IPSP ,N);
                                    ALOSS[L]=ALOSS[L]+1.e00;
                                }
                                REMOVE_MOL(N);
                                N=N-1;
                                II=1;
                            }
                            //
                            if(ITYPE[J] == 1){
                                if((IVB == 0) || (J == 1)){
                                    X=2.e00*XB[J]-X;
                                    get(PV  ,1,N)=-get(PV  ,1,N);
                                }
                                else if((J == 2) && (IVB == 1)){
                                    DTC=(XB[2]+TI*VELOB-XI)/(get(PV  ,1,N)-VELOB);
                                    XC=XI+get(PV  ,1,N)*DTC;
                                    get(PV  ,1,N)=-get(PV  ,1,N)+2*VELOB;
                                    X=XC+get(PV  ,1,N)*(DTIM-DTC);
                                }
                            }
                            //
                            if(ITYPE[J] == 2)
                                REFLECT_1D(N,J,X);
                            // END IF
                        }
                    }
                }
            }
            else{         //cylindrical or spherical flow
                //check boundaries
                if((X < XB[1]) && (XB[1] > 0.e00)){
                    RBC(XI,DX,DY,DZ,XB[1],S1);
                    if(S1 < 1.e00){     //intersection with inner boundary
                        if(ITYPE[1] == 2){//solid surface
                            DX=S1*DX;
                            DY=S1*DY;
                            DZ=S1*DZ;
                            AIFX(XI,DX,DY,DZ,X,get(PV  ,1,N),get(PV  ,2,N),get(PV  ,3,N));
                            REFLECT_1D(N,1,X);
                        }
                        else{
                            REMOVE_MOL(N);
                            N=N-1;
                            II=1;
                        }
                    }
                }
                else if((IVB == 0) && (R > XB[2])){
                    RBC(XI,DX,DY,DZ,XB[2],S1);
                    if(S1 < 1.e00){     //intersection with outer boundary
                        if(ITYPE[2] == 2){ //solid surface
                            DX=S1*DX;
                            DY=S1*DY;
                            DZ=S1*DZ;
                            AIFX(XI,DX,DY,DZ,X,get(PV  ,1,N),get(PV  ,2,N),get(PV  ,3,N));
                            X=1.001e00*XB[2];
                            while(X > XB[2])
                                REFLECT_1D(N,2,X);
                            // END DO
                        }
                        else{
                            REMOVE_MOL(N);
                            N=N-1;
                            II=1;
                        }
                    }
                }
                else if((IVB == 1) && (R > (XB[2]+PTIM[N]*VELOB))){
                    if(IFX == 1) UR=sqrt(pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2));
                    if(IFX == 2) UR=sqrt(pow(get(PV  ,1,N),2)+pow(get(PV  ,2,N),2)+pow(get(PV  ,3,N),2));
                    DTC=(XB[2]+TI*VELOB-XI)/(UR-VELOB);
                    S1=DTC/DTIM;
                    DX=S1*DX;
                    DY=S1*DY;
                    DZ=S1*DZ;
                    AIFX(XI,DX,DY,DZ,X,get(PV  ,1,N),get(PV  ,2,N),get(PV  ,3,N));
                    get(PV  ,1,N)=-get(PV  ,1,N)+2.0*VELOB;
                    X=X+get(PV  ,1,N)*(DTIM-DTC);
                }
                else
                    AIFX(XI,DX,DY,DZ,X,get(PV  ,1,N),get(PV  ,2,N),get(PV  ,3,N));
                
                
                //DIAGNOSTIC
                if(II == 0){
                    if(X > XB[2]+PTIM[N]*VELOB){
                        //WRITE (*,*) N,FTIME,X,XB[2]+PTIM[N]*VELOB;
                        cout<<N<<" "<<FTIME<<" "<<X<<" "<<(XB[2]+PTIM[N]*VELOB)<<endl;
                    }
                }
                
                //Take action on weighting factors
                if((IWF == 1) && (II == 0)){
                    WFR=WFI/(1.e00+WFM*pow(X,IFX));
                    L=0;
                    WFRI=WFR;
                    if(WFR >= 1.e00){
                        while(WFR >= 1.e00){
                            L=L+1;
                            WFR=WFR-1.e00;
                        }
                    }
                    // CALL RANDOM_NUMBER(RANF)
                    RANF=((double)rand()/(double)RAND_MAX);
                    if(RANF <= WFR) L=L+1;
                    if(L == 0){
                        REMOVE_MOL(N);
                        N=N-1;
                        II=1;
                    }
                    L=L-1;
                    if(L > 0){
                        for(K=1;K<=L;K++){
                            if(NM >= MNM) EXTEND_MNM(1.1);
                            NM=NM+1;
                            get (PX , 1,NM)=X;
                            for(M=1;M<=3;M++)
                                get(PV  ,M,NM)=get(PV  ,M,N);
                            
                            if(MMRM > 0) PROT[NM]=PROT[N];
                            get(IPCELL  ,NM)=fabs(get(IPCELL  ,N));
                            get(IPSP ,NM)=get(IPSP ,N);
                            IPCP[NM]=IPCP[N];
                            if(MMVM > 0){
                                for(M=1;M<=MMVM;M++)
                                    get (IPVIB , M,NM)=get (IPVIB , M,N);
                                
                            }
                            PTIM[NM]=PTIM[N];    //+5.D00*DFLOAT(K)*DTM
                            //note the possibility of a variable time advance that may take the place of the duplication buffer in earlier programs
                            
                            if(get (PX , 1,NM) > XB[2]+PTIM[NM]*VELOB)
                                //WRITE (*,*) 'DUP',NM,FTIME,PX(1,NM),XB(2)+PTIM(NM)*VELOB
                                cout<<"DUP "<<NM<<" "<<FTIME<<" "<<get (PX , 1,NM)<<" "<<(XB[2]+PTIM[NM]*VELOB)<<endl;
                            
                        }
                    }
                }
            }
            //
            if(II == 0) {
                get (PX , 1,N)=X;
                
                if(get (PX , 1,N) > XB[1] && (get (PX , 1,N) < XB[2]))
                    continue;
                else{
                    //cout<< N<<" OUTSIDE FLOWFIELD AT "<<get (PX , 1,N]<<" VEL "<<get(PV  ,1,N]<<endl;
                    REMOVE_MOL(N);
                    N=N-1;
                    II=1;
                }
            }
            //
            if(II == 0){
                if(IVB == 0) FIND_CELL_1D(get (PX , 1,N),get(IPCELL  ,N),JJ);
                if(IVB == 1) FIND_CELL_MB_1D(get (PX , 1,N),get(IPCELL  ,N),JJ,PTIM[N]);
            }
            //
        }
        //
        N=N+1;
    }
    //
    return;
}


void MOLECULES_ENTER_1D()
{
    //molecules enter boundary at XB(1) and XB(2) and may be removed behind a wave
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //GEOM_1D geom;

    //
    int K,L,M,N,NENT,II,J,JJ,KK,NTRY;
    double A,B,AA,BB,U,VN,XI,X,DX,DY,DZ;
    //
    //NENT number to enter in the time step
    //
    ENTMASS=0.e00;
    //
    for(J=1;J<=2;J++){     //J is the end
        if((ITYPE[J] == 0) || (ITYPE[J] == 4)){
            KK=1;//the entry surface will normally use the reference gas (main stream) properties
            if((J == 2) && (ISECS == 1) && (XB[2] > 0.e00)) KK=2;    //KK is 1 for reference gas 2 for the secondary stream
            for(L=1;L<=MSP;L++){
                A=get (ENTR , 1,L,J)*DTM+get (ENTR , 2,L,J);
                if((ITYPE[2] == 4) && (ICN == 1)){
                    NENT=A;
                    if(J == 1) EME[L]=NENT;
                    if(J == 2) {
                        A=ALOSS[L]-EME[L]-AJM[L];
                        AJM[L]=0.e00;
                        if(A < 0.e00){
                            AJM[L]=-A;
                            A=0.e00;
                        }
                    }
                }
                NENT=A;
                get (ENTR , 2,L,J)=A-NENT;
                if((ITYPE[2] == 4) && (J == 2) && (ICN == 1)) get (ENTR , 2,L,J)=0.e00;
                if(NENT > 0){
                    for(M=1;M<=NENT;M++){
                        if(NM >= MNM){
                          
                            EXTEND_MNM(1.1);
                        }
                        NM=NM+1;
                        AA=max(0.e00,get (ENTR , 3,L,J)-3.e00);
                        BB=max(3.e00,get (ENTR , 3,L,J)+3.e00);
                        II=0;
                        while(II == 0){
                            RANF=((double)rand()/(double)RAND_MAX);
                            // CALL RANDOM_NUMBER(RANF)
                            B=AA+(BB-AA)*RANF;
                            U=B-get (ENTR , 3,L,J);
                            A=(2.e00*B/get (ENTR , 4,L,J))*exp(get (ENTR , 5,L,J)-U*U);
                            RANF=((double)rand()/(double)RAND_MAX);
                            // CALL RANDOM_NUMBER(RANF)
                            if(A > RANF) II=1;
                        }
                        get(PV  ,1,NM)=B*get (VMP , L,KK);
                        if(J == 2) get(PV  ,1,NM)=-get(PV  ,1,NM);
                        //
                        RVELC(get(PV  ,2,NM),get(PV  ,3,NM),get (VMP , L,KK));
                        get(PV  ,2,NM)=get(PV  ,2,NM)+VFY[J];
                        //
                        if(get(ISPR ,1,L) > 0) SROT(L,FTMP[KK],PROT[NM]);
                        //
                        if(MMVM > 0){
                            for(K=1;K<=get(ISPV  ,L);K++)
                                SVIB(L,FVTMP[KK],get (IPVIB , K,NM),K);
                        }
                        if(MELE > 1) SELE(L,FTMP[KK],PELE[NM]);
                        //
                        if(PELE[NM] > 0.e00)
                            continue;                     //DEBUG
                        //
                        get(IPSP ,NM)=L;
                        //advance the molecule into the flow
                        RANF=((double)rand()/(double)RAND_MAX);
                        // CALL RANDOM_NUMBER(RANF)
                        XI=XB[J];
                        DX=DTM*RANF*get(PV  ,1,NM);
                        if((IFX == 0) || (J == 2)) X=XI+DX;
                        if(J == 1){   //1-D move at outer boundary so molecule remains in flow
                            if(IFX > 0) DY=DTM*RANF*get(PV  ,2,NM);
                            DZ=0.e00;
                            if(IFX == 2) DZ=DTM*RANF*get(PV  ,3,NM);
                            if(IFX > 0) AIFX(XI,DX,DY,DZ,X,get(PV  ,1,NM),get(PV  ,2,NM),get(PV  ,3,NM));
                        }
                        get (PX , NCLASS,NM)=X;
                        PTIM[NM]=FTIME;
                        if(IVB == 0) FIND_CELL_1D(get (PX , NCLASS,NM),get(IPCELL  ,NM),JJ);
                        if(IVB == 1) FIND_CELL_MB_1D(get (PX , NCLASS,NM),get(IPCELL  ,NM),JJ,PTIM[NM]);
                        IPCP[NM]=0;
                        if(XREM > XB[1]) ENTMASS=ENTMASS+get(SP ,5,L);
                    }
                }
            }
            if((ITYPE[2] == 4) && (J==2) && (NM != NMP) && (ICN == 1))
                continue;
        }
    }
    //
    //stagnation streamline molecule removal
    if(XREM > XB[1]){
        ENTMASS=FREM*ENTMASS;
        NTRY=0;
        ENTMASS=ENTMASS+ENTREM;
        while((ENTMASS > 0.e00) && (NTRY < 10000)){
            NTRY=NTRY+1;
            if(NTRY == 10000){
                cout<<"Unable to find molecule for removal"<<endl;
                ENTMASS=0.e00;
                //memget(VNMAX,0.e00,sizeof(*VNMAX));//VNMAX=0.e00;
                for(int i=0;i<MSP+1;i++)
                    VNMAX[i]=0.e00;
            }
            RANF=((double)rand()/(double)RAND_MAX) ;
            // CALL RANDOM_NUMBER(RANF)
            N=NM*RANF+0.9999999e00;
            if(get (PX , NCLASS,N) > XREM){
                // CALL RANDOM_NUMBER(RANF)
                RANF=((double)rand()/(double)RAND_MAX) ;
                //IF (RANF < ((PX(N)-XREM)/(XB(2)-XREM))*2) THEN
                if(fabs(VFY[1]) < 1.e-3)
                    VN=sqrt(get(PV  ,2,N)*get(PV  ,2,N)+get(PV  ,3,N)*get(PV  ,3,N)) ;   //AXIALLY SYMMETRIC STREAMLINE
                else
                    VN=fabs(get(PV  ,3,N)) ;   //TWO-DIMENSIONAL STREAMLINE
                 
                L=get(IPSP ,N);
                if(VN > VNMAX[L]) VNMAX[L]=VN;
                // CALL RANDOM_NUMBER(RANF)
                RANF=((double)rand()/(double)RAND_MAX);
                if(RANF < VN/VNMAX[L]){
                    REMOVE_MOL(N);
                    ENTMASS=ENTMASS-get(SP ,5,L);
                    NTRY=0;
                }
                //END IF
            }
        }
        ENTREM=ENTMASS;
    }
}


void INDEX_MOLS()
{
    //index the molecules to the collision cells
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    // IMPLICIT NONE
    //
    int N,M,K;
    //
    //N,M,K working integer
    //
    for(N=0 ;N<=NCCELLS;N++)
        get (ICCELL , 2,N)=0;

    
    //
    if(NM != 0){
        for(N=1;N<=NM;N++){
            M=get(IPCELL  ,N);
            get (ICCELL , 2,M)=get (ICCELL , 2,M)+1;
        }
        //

        M=0;
        for(N=1;N<=NCCELLS;N++){
            get (ICCELL , 1,N)=M;
            M=M+get (ICCELL , 2,N);
            get (ICCELL , 2,N)=0;
        }
        //

        for(N=1;N<=NM;N++){
            M=get(IPCELL  ,N);
            get (ICCELL , 2,M)=get (ICCELL , 2,M)+1;
            K=get (ICCELL , 1,M)+get (ICCELL , 2,M);
            ICREF[K]=N;
        }
        //cin.get();
        //
    }
    return;
}


void RBC(double &XI, double &DX, double &DY,double &DZ, double &R,double &S)
{
    //calculates the trajectory fraction S from a point at radius XI with
    //note that the axis is in the y direction
    //--displacements DX, DY, and DZ to a possible intersection with a
    //--surface of radius R, IFX=1, 2 for cylindrical, spherical geometry
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    double A,B,C,DD,S1,S2;
    //
    DD=DX*DX+DZ*DZ;
    if(IFX == 2) DD=DD+DY*DY;
    B=XI*DX/DD;
    C=(XI*XI-R*R)/DD;
    A=B*B-C;
    if(A >= 0.e00){
        //find the least positive solution to the quadratic
        A=sqrt(A);
        S1=-B+A;
        S2=-B-A;
        if(S2 < 0.e00){
            if(S1 > 0.e00)
                S=S1;
            else
                S=2.e00;
        }
        else if(S1 < S2)
            S=S1;
        else
            S=S2;
    }
    else
        S=2.e00;
    //setting S to 2 indicates that there is no intersection
    return;
    //
}

void AIFX(double &XI,double &DX, double &DY, double &DZ, double &X, double &U, double &V, double &W)
{
    //
    //calculates the new radius and realigns the velocity components in
    //--cylindrical and spherical flows
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    //INTEGER ::
    double A,B,C,DR,VR,S;
    //
    if(IFX == 1){
        DR=DZ;
        VR=W;
    }
    else if(IFX == 2){
        DR=sqrt(DY*DY+DZ*DZ);
        VR=sqrt(V*V+W*W);
    }
    A=XI+DX;
    X=sqrt(A*A+DR*DR);
    S=DR/X;
    C=A/X;
    B=U;
    U=B*C+VR*S;
    W=-B*S+VR*C;
    if(IFX == 2){
        VR=W;
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        A=DPI*RANF;
        V=VR*sin(A);
        W=VR*cos(A);
    }
    //
    return;
    //
}


void REMOVE_MOL(int &N)
{
    //remove molecule N and replaces it by NM
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    // IMPLICIT NONE
    //
    int NC,M,K;
    
    //N the molecule number
    //M,K working integer
    //
    if(N != NM){
        for(M=1;M<=NCLASS;M++)
            get (PX , M,N)=get (PX , M,NM);
        for(M=1;M<=3;M++)
            get(PV  ,M,N)=get(PV  ,M,NM);
        
        if(MMRM > 0) PROT[N]=PROT[NM];
        get(IPCELL  ,N)=fabs(get(IPCELL  ,NM));
        get(IPSP ,N)=get(IPSP ,NM);
        IPCP[N]=IPCP[NM];
        if(MMVM > 0){
            for(M=1;M<=MMVM;M++)
                get (IPVIB , M,N)=get (IPVIB , M,NM);
        }
        if(MELE > 1) PELE[N]=PELE[NM];
        PTIM[N]=PTIM[NM];
    }
    NM=NM-1;
    //
    return;
    //
}

void DISSOCIATION()
{
    //dissociate diatomic molecules that have been marked for dissociation by -ve level or -99999 for ground state
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,KK,L,N,M,LS,MS,KV,IDISS;
    double A,B,C,EA,VRR,VR,RMM,RML;
    double VRC[4],VCM[4],VRCP[4];
    //
    N=0;
    while(N < NM){
        N=N+1;
        IDISS=0;
        L=get(IPSP ,N);
        if(get(ISPV  ,L) > 0){
            for(K=1;K<=get(ISPV  ,L);K++){
                M=get (IPVIB , K,N);
                if(M < 0){
                    //dissociation
                    TDISS[L]=TDISS[L]+1.e00;
                    IDISS=1;
                }
            }
            if(IDISS == 1){
                EA=PROT[N];    //EA is energy available for relative translational motion of atoms
                if(MELE > 1) EA=EA+PELE[N];
                if(NM >= MNM) EXTEND_MNM(1.1);
                NM=NM+1;
                //set center of mass velocity as that of molecule
                VCM[1]=get(PV  ,1,N);
                VCM[2]=get(PV  ,2,N);
                VCM[3]=get(PV  ,3,N);
                get (PX , NCLASS,NM)=get (PX , NCLASS,N);
                get(IPCELL  ,NM)=get(IPCELL  ,N);
                LS=get(IPSP ,N);
                get (TREACL , 1,LS)=get (TREACL , 1,LS)-1;
                get(IPSP ,NM)=get( ISPVM ,1,1,L);
                MS=get(IPSP ,NM);
                get(IPSP ,N)=get( ISPVM ,2,1,L);
                LS=get(IPSP ,N);
                get (TREACG , 1,LS)=get (TREACG , 1,LS)+1;
                get (TREACG , 1,MS)=get (TREACG , 1,MS)+1;
                PTIM[NM]=PTIM[N];
                VRR=2.e00*EA/get (SPM , 1,LS,MS);
                VR=sqrt(VRR);
                RML=get (SPM , 1,LS,MS)/get(SP ,5,MS);
                RMM=get (SPM , 1,LS,MS)/get(SP ,5,LS);
                // CALL RANDOM_NUMBER(RANF)
                RANF=((double)rand()/(double)RAND_MAX);
                B=2.e00*RANF-1.e00;
                A=sqrt(1.e00-B*B);
                VRCP[1]=B*VR;
                // CALL RANDOM_NUMBER(RANF)
                RANF=((double)rand()/(double)RAND_MAX);
                C=2.e00*PI*RANF;
                VRCP[2]=A*cos(C)*VR;
                VRCP[3]=A*sin(C)*VR;
                for(KK=1;KK<=3;KK++){
                    get(PV  ,KK,N)=VCM[KK]+RMM*VRCP[KK];
                    get(PV  ,KK,NM)=VCM[KK]-RML*VRCP[KK];
                }
                
                if((fabs(get(PV  ,1,N)) > 100000.e00) || (fabs(get(PV  ,1,NM)) > 100000.e00)) {
                    cout<< "EXCESSIVE SPEED, DISS "<< N<< " "<<get(PV  ,1,N)<<" "<<NM<<" "<<get(PV  ,1,NM)<<endl;
                   
                }
                
                
                
                //set any internal modes to the ground state
                if(get(ISPV  ,LS) > 0){
                    for(KV=1;KV<=get(ISPV  ,LS);KV++)
                        get (IPVIB , KV,N)=0;
                }
                if(get(ISPR ,1,LS) > 0) PROT[N]=0.e00;
                if(MELE > 1) PELE[N]=0.e00;
                if(get(ISPV  ,MS) > 0){
                    for(KV=1;KV<=get(ISPV  ,MS);KV++)
                        get (IPVIB , KV,NM)=0;
                }
                if(get(ISPR ,1,MS) > 0) PROT[NM]=0.0;
                if(MELE > 1) PELE[NM]=0.e00;
            }
        }
    }
    return;
}

void EXTEND_MNM(double FAC)
{  //
    //the maximum number of molecules is increased by a specified factor
    //the existing molecules are copied TO disk storage
    //MOLECS molecs;
    //CALC calc;
    //GAS gas;
    //
    // IMPLICIT NONE
    //
    int M,N,MNMN;
    fstream file_7;
    // REAL :: FAC
    //
    //M,N working integers
    //MNMN extended value of MNM
    //FAC the factor for the extension
    MNMN=FAC*MNM;
    cout<< "Maximum number of molecules is to be extended from "<<MNM<<" to "<<MNMN<<endl;
    cout<< "( if the additional memory is available //// )"<<endl;
    
    file_7.open("EXTMOLS.TXT", ios::binary | ios::out);
    if(file_7.is_open()){
        cout<<"EXTMOLS.TXT is opened"<<endl;
    }
    else{
        cout<<"EXTMOLS.TXT not opened"<<endl;
    }
    cout<<"Start write to disk storage"<<endl;
    //OPEN (7,FILE='EXTMOLS.SCR',FORM='BINARY')
    //WRITE (*,*) 'Start write to disk storage'
    
    for(N=1;N<=MNM;N++){
        if(MMVM > 0){
            file_7<<get (PX , NCLASS,N)<<endl<<PTIM[N]<<endl<<PROT[N]<<endl;
            for(M=1;M<=3;M++)
                file_7<<get(PV  ,M,N)<<endl;
            file_7<<get(IPSP ,N)<<endl<<get(IPCELL  ,N)<<endl<<ICREF[N]<<endl<<IPCP[N]<<endl;
            for(M=1;M<=MMVM;M++)
                file_7<<get (IPVIB , M,N)<<endl;
            file_7<<PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),(get(IPVIB  ,M,N),M=1,MMVM),PELE(N)
        }
        else{
            if(MMRM > 0){
                file_7<<get (PX , NCLASS,N)<<endl<<PTIM[N]<<endl<<PROT[N]<<endl;
                for(M=1;M<=3;M++)
                    file_7<<get(PV  ,M,N)<<endl;
                file_7<<get(IPSP ,N)<<endl<<get(IPCELL  ,N)<<endl<<ICREF[N]<<endl<<IPCP[N]<<endl<<PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            else{
                file_7<<get (PX , NCLASS,N)<<endl<<PTIM[N]<<endl;
                for(M=1;M<=3;M++)
                    file_7<<get(PV  ,M,N)<<endl;
                file_7<<get(IPSP ,N)<<endl<<get(IPCELL  ,N)<<endl<<ICREF[N]<<endl<<IPCP[N]<<endl<<PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            
        }
    }
    cout<<"Disk write completed"<<endl;
    // WRITE (*,*) 'Disk write completed'
    // CLOSE (7)
    file_7.close();
    if(MMVM > 0){
        
        free(PX); //delete [) PX;

        free(PTIM); //delete [) PTIM;

        free(PROT);

        
        free(PV); //delete [) PV;

        free(IPSP);
        free(IPCELL);
        free(ICREF);
        free(IPCP);
        free(PELE);
        
        free(IPVIB); //delete IPVIB;
        // for(int i=0;i<NCLASS+1;i++){
        //     delete [) get (PX , i];
        // }
        // delete [] PX;
        // delete [] PTIM;
        // delete [] PROT;
        // for(int i=0;i<4;i++){
        //     delete [] get(PV  ,i];
        // }
        // delete [] PV;
        // delete [] IPSP;
        // delete [] IPCELL;
        // delete [] ICREF;
        // delete [] IPCP;
        // delete [] PELE;
        // for(int i=0;i<MMVM;i++){
        //     delete [] get (IPVIB , i];
        // }
        // delete IPVIB;
        //DEALLOCATE (PX,PTIM,PROT,PV,IPSP,IPCELL,ICREF,IPCP,IPVIB,PELE,STAT=ERROR)
    }
    else{
        if(MMRM > 0){
            
            free(PX); //delete [) PX;

            free(PTIM); //delete [) PTIM;

            free(PROT);

           
            free(PV); //delete [) PV;

            free(IPSP);
            free(IPCELL);
            free(ICREF);
            free(IPCP);
            free(PELE);
            // delete [) IPSP;
            // delete [) IPCELL;
            // delete [) ICREF;
            // delete [) IPCP;
            // delete [) PELE;//DEALLOCATE (PX,PTIM,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
            // for(int i=0;i<NCLASS+1;i++){
            //     delete [) get (PX , i);
            // }
            // delete [) PX;
            // delete [) PTIM;
            // delete [) PROT;
            // for(int i=0;i<4;i++){
            //     delete [) get(PV  ,i);
            // }
            // delete [) PV;
            // delete [) IPSP;
            // delete [) IPCELL;
            // delete [) ICREF;
            // delete [) IPCP;
            // delete [) PELE;
            //DEALLOCATE (PX,PTIM,PROT,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
        }
        else{
            
            free(PX); //delete [) PX;

            free(PTIM); //delete [) PTIM;

          
            free(PV); //delete [) PV;

            free(IPSP);
            free(IPCELL);
            free(ICREF);
            free(IPCP);
            free(PELE);
            // delete [) IPSP;
            // delete [) IPCELL;
            // delete [) ICREF;
            // delete [) IPCP;
            // delete [) PELE;//DEALLOCATE (PX,PTIM,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT DEALLOCATE MOLECULES',ERROR
    // !  STOP
    // END IF
    // !
    
    if(MMVM > 0){
        d_allocate(NCLASS,MNMN,PX);
        d_allocate(MNMN,PTIM);
        d_allocate(MNMN,PROT);
        d_allocate(3,MNMN,PV);
        i_allocate(MNMN,IPSP);
        i_allocate(MNMN,IPCELL);
        i_allocate(MNMN,ICREF);
        i_allocate(MNMN,IPCP);
        i_allocate(MMVM,MNMN,IPVIB);
        d_allocate(MNMN,PELE);
        // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PROT(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),get(IPVIB  ,MMVM,MNMN),PELE(MNMN),STAT=ERROR)
    }
    else{
        if(MMRM > 0){
            d_allocate(NCLASS,MNMN,PX);
            d_allocate(MNMN,PTIM);
            d_allocate(MNMN,PROT);
            d_allocate(3,MNMN,PV);
            i_allocate(MNMN,IPSP);
            i_allocate(MNMN,IPCELL);
            i_allocate(MNMN,ICREF);
            i_allocate(MNMN,IPCP);
            d_allocate(MNMN,PELE);
            // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PROT(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),PELE(MNMN),STAT=ERROR)
        }
        else{
            d_allocate(NCLASS,MNMN,PX);
            d_allocate(MNMN,PTIM);
            d_allocate(3,MNMN,PV);
            i_allocate(MNMN,IPSP);
            i_allocate(MNMN,IPCELL);
            i_allocate(MNMN,ICREF);
            i_allocate(MNMN,IPCP);
            d_allocate(MNMN,PELE);
            // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),PELE(MNMN),STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SPACE FOR EXTEND_MNM',ERROR
    // !  STOP
    // END IF
    // !
    //memget(PX,0.0,sizeof(*PX)); memget(PTIM,0.0,sizeof(*PTIM)); memget(PV,0.0,sizeof(*PV)); memget(IPSP,0,sizeof(*IPSP)); memget(IPCELL,0,sizeof(*IPCELL)); memget(ICREF,0,sizeof(*ICREF)); memget(IPCP,0,sizeof(*IPCP)); memget(PELE,0,sizeof(*PELE));
    
    for(int i=0;i<NCLASS+1;i++){
        for(int j=0;j<MNMN+1;j++)
            get (PX , i,j)=0.0;
    }
    
    for(int i=0;i<4;i++){
        for(int j=0;j<MNMN+1;j++)
            get(PV  ,i,j)=0.0;
    }
    for(int i=0;i<MNMN+1;i++){
        PTIM[i]=0.0;
        get(IPSP ,i)=0;
        get(IPCELL  ,i)=0;
        ICREF[i]=0;
        IPCP[i]=0;
        PELE[i]=0;
    }
        
    
    if(MMRM > 0) {
        for(int i=0;i<MNMN+1;i++)
            PROT[i]=0.0;
        //memget(PROT,0.0,sizeof(*PROT));
    }
    if(MMVM > 0) {
        for(int i=0;i<MMVM+1;i++){
            for(int j=0;j<MNMN+1;j++)
                get (IPVIB , i,j)=0;
        }
        //memget(IPVIB,0,sizeof(*IPVIB));
    }
    //restore the original molecules
    // OPEN (7,FILE='EXTMOLS.SCR',FORM='BINARY')
    // WRITE (*,*) 'Start read back from disk storage'
    file_7.open("EXTMOLS.TXT", ios::binary | ios::in);
    if(file_7.is_open()){
        cout<<"EXTMOLS.TXT is opened"<<endl;
    }
    else{
        cout<<"EXTMOLS.TXT not opened"<<endl;
    }
    for(N=1;N<=MNM;N++){
        if(MMVM > 0){
            file_7>>get (PX , NCLASS,N)>>PTIM[N]>>PROT[N];
            for(M=1;M<=3;M++)
                file_7>>get(PV  ,M,N);
            file_7>>get(IPSP ,N)>>get(IPCELL  ,N)>>ICREF[N]>>IPCP[N];
            for(M=1;M<=MMVM;M++)
                file_7>>get (IPVIB , M,N);
            file_7>>PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),(get(IPVIB  ,M,N),M=1,MMVM),PELE(N)
        }
        else{
            if(MMRM > 0){
                file_7>>get (PX , NCLASS,N)>>PTIM[N]>>PROT[N];
                for(M=1;M<=3;M++)
                    file_7>>get(PV  ,M,N);
                file_7>>get(IPSP ,N)>>get(IPCELL  ,N)>>ICREF[N]>>IPCP[N]>>PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            else{
                file_7>>get (PX , NCLASS,N)>>PTIM[N];
                for(M=1;M<=3;M++)
                    file_7>>get(PV  ,M,N);
                file_7>>get(IPSP ,N)>>get(IPCELL  ,N)>>ICREF[N]>>IPCP[N]>>PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
        }
    }
    cout<<"Disk read completed"<<endl;
    // WRITE (*,*) 'Disk read completed'
    // CLOSE (7,STATUS='DELETE')
    file_7.close();
    //
    MNM=MNMN;
    //
    return;
}

void SAMPLE_FLOW()
{
    //sample the flow properties
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    file_3 << "SAMPLE_FLOW IS Running \n" ;
    file_9 << "sample flow is running\n" ;
    //GAS gas;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int NC,NCC,LS,N,M,K,L,I,KV;
    double A,TE,TT,WF;
    //
    //NC the sampling cell number
    //NCC the collision cell number
    //LS the species code
    //N,M,K working integers
    //TE total translational energy
    //
    NSAMP=NSAMP+1;
    cout<<"Sample \t"<<NSAMP<<endl<<endl;
    //WRITE (9,*) NM,'Mols. at sample',NSAMP
    file_9<<NM<<"\t  Mols. at sample \t "<<NSAMP<<endl;
    //
    cout << "get(PV  ,1,4856) = "<< get(PV  ,1,4856) << endl ;
    for(N=1;N<=NM;N++){
        
        //if(N == 4856)   continue ;  // in cpp you will see that this number is equal to infinity.
        NCC=get(IPCELL  ,N);
        NC=get (ICCELL , 3,NCC);
        WF=1.e00;
        if(IWF == 1) WF=1.e00+WFM*pow(get (PX , 1,N),IFX);
        if((NC > 0) && (NC <= NCELLS)){
            if(MSP > 1)
                LS=fabs(get(IPSP ,N));
            else
                LS=1;
            
            get (CS,1+ 0,NC,LS)=get (CS,1+ 0,NC,LS)+1.e00;
            get (CS,1+ 1,NC,LS)=get (CS,1+ 1,NC,LS)+WF;
            if(N == 1)
            
            for(M=1;M<=3;M++){
                get (CS,1+ M+1,NC,LS) = get (CS,1+ M+1,NC,LS) + WF*get(PV  ,M,N) ;
                get (CS,1+ M+4,NC,LS)=get (CS,1+ M+4,NC,LS)+ WF*pow(get(PV  ,M,N),2) ;
            }
            if(MMRM > 0) get (CS,1+ 8,NC,LS)=get (CS,1+ 8,NC,LS)+WF*PROT[N];
            if(MELE > 1) get (CS,1+ 9,NC,LS)=get (CS,1+ 9,NC,LS)+WF*PELE[N];
            if(MMVM > 0){
                if(get(ISPV  ,LS) > 0){
                    for(K=1;K<=get(ISPV  ,LS);K++)
                        get (CS,1+ K+9,NC,LS)=get (CS,1+ K+9,NC,LS)+WF*(double)(get (IPVIB , K,N));
                }
            }
        }
        else{
            cout<<"Illegal sampling cell  "<<NC<<"  "<<NCC<<"  for MOL  "<<N<<"  at  "<<get (PX , 1,N)<<endl;
            return;
        }


        
    }

    for(int II = 0 ; II <11 ; II++)
        {
            cout << "get (CS,1+ "<<II<<",1,1] =" <<get (CS, 1+ II , 1, 1 ) <<endl ;
        }
    //
    
    if(FTIME > 0.5e00*DTM) TSAMP=TSAMP+DTSAMP;
    //
    return;
}

string itos(int c)
{
    stringstream ss ;
    ss << c  ;
    string b= ss.str() ;
    return b ;
}

void OUTPUT_RESULTS()
{
    //--calculate the surface and flowfield properties
    //--generate TECPLOT files for displaying these properties
    //--calculate collisiion rates and flow transit times and reset time intervals
    //--add molecules to any flow plane molecule output files
    //CALC calc;
    //MOLECS molecs;
    //GAS gas;
    //OUTPUT output;
    //GEOM_1D geom;
    cout << 111 << endl ;
    file_9 << "output results is running \n" ;
    fstream file_3;
    fstream file_10;
    fstream file_7;
    
    int IJ,J,JJ,K,L,LL,M,N,NN,NMCR,CTIME,II;
    long long NNN;
    double AS,AT,C1,C2,C3,C4,C5,C6,C7,C8,C9;
    double A,B,C,SDTM,SMCR,DOF,AVW,UU,VDOFM,TVIBM,VEL,DTMI,TT;
    //dout
    double SUM[14];
    //d_allocate(14 , SUM) ;
    double *SUMS;
    d_allocate( 10 , 2 , SUMS ) ;
    double *TVIB,*VDOF,*PPA,*TEL,*ELDOF,*SDOF,*CDTM;
    double *TV,*THCOL;
    double *DF;
    int *NMS;
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:) :: TVIB,VDOF,PPA,TEL,ELDOF,SDOF,CDTM
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: TV,THCOL
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:,:) :: DF
    //    INTEGER, ALLOCATABLE, DIMENSION(:) :: NMS
    //INTEGER, ALLOCATABLE, DIMENSION(:,:) ::
    string F,E;
    //--CTIME  computer time (microseconds)
    //--SUMS(N,L) sum over species of CSS(N,J,L,M) for surface properties
    //
    //--For flowfield properties,where <> indicates sampled sum
    //--SUM(0) the molecular number sum over all species
    //--SUM(1) the weighted number sum over all species
    //--SUM(2) the weighted sum of molecular masses
    //--SUM(3),(4),(5) the weighted sum over species of m*<u>,<v>,<w>
    //--SUM(6) the weighted sum over species of m*(<u*2>+<v*2>+<w*2>)
    //--SUM(7) the weighted sum over species of <u*2>+<v*2>+<w*2>
    //--SUM(8) the weighted sum of rotational energy
    //--SUM(9) the weighted sum of rotational degrees of freedom
    //--SUM(10) the weighted sum over species of m*<u*2>
    //--SUM(11) the weighted sum over species of m*<v*2>
    //--SUM(12) sum over species of m*<w*2>
    //--SUM(13) the weighted sum of electronic energy
    //--UU velocity squared
    //--DOF degrees of freedom
    //--AVW the average value of the viscosity-temperature exponent
    //--DVEL velocity difference
    //--TVEL thermal speed
    //--SMCR sum of mcs/mfp over cells
    //--NMCR number in the sum
    //--VDOFM effective vibrational degrees of freedom of mixture
    //--TVIB(L)
    //--VDOF(L)
    //--TV(K,L) the temperature of vibrational mode K of species L
    //--PPA particles per atom
    //--NMS number per species
    //--SDOF(L) total degrees of freedom for species L
    //
    //
    //--calculate the flowfield properties in the cells
    //dout
    
    d_allocate(MMVM , MSP, TV ) ;
    
    d_allocate(MSP, TVIB) ;
    
    
    d_allocate(NCELLS , MMVM , MSP  , DF) ;
    
    //VDOF= new double[MSP];
    d_allocate(MSP, VDOF );
    //TEL = new double[MSP];
    d_allocate(MSP, TEL) ;
    //ELDOF = new double[MSP];
    d_allocate(MSP, ELDOF) ;
    //PPA = new double[MSP];
    d_allocate(MSP, PPA) ;
    //NMS = new int[MSP];
    i_allocate(MSP, NMS) ;
    
    d_allocate(MSP, MSP, THCOL) ;
    
    d_allocate(MSP, SDOF) ;
    
    d_allocate(NCELLS , CDTM) ;
    
    
    //    ALLOCATE (TV(MMVM,MSP),TVIB(MSP),DF(NCELLS,MMVM,MSP),VDOF(MSP),TEL(MSP),ELDOF(MSP),PPA(MSP),NMS(MSP),THCOL(MSP,MSP)    &
    //              ,SDOF(MSP),CDTM(NCELLS),STAT=ERROR)
    //    if(ERROR!=0)
    //    {
    //        cout<<"ROGRAM COULD NOT ALLOCATE OUTPUT VARIABLES"<<ERROR<<endl;
    //    }
    if(FTIME>0.5e00*DTM)
    {
        NOUT+=1;
        if(NOUT>9999)
            NOUT=NOUT-9999;
        cout << "ISF = "  << ISF << endl ;  // dsuedit
        cout<<"Generating files for output interval"<<NOUT<<endl;
        if(ISF==0)
        {
            //dout
            //OPEN (3,FILE='DS1OUT.DAT')
            file_3.open("DS1OUT.DAT" , ios::out);
            if(file_3.is_open()){
                cout<<"DS1OUT.DAT is opened"<<endl;
            }
            else{
                cout<<"DS1OUT.DAT not opened"<<endl;
            }
            //F='DS';//E//'.OUT'
        }
        else
        {
            //--the files are DS1n.DAT, where n is a four digit integer equal to NOUT
            //dout
            //500 FORMAT(I5)
            //ENCODE(5,500,E) 10000+NOUT

            cout << "ISF = "<< ISF << endl ;

            int a=NOUT+10000;
            E= itos(a) ;
            F="DS" + E + "OUT.DAT";

    // copying the contents of the  
    // string to char array 
            //strcpy(char_array, F.c_str()); 
            //dout
            file_3.open(F.c_str() , ios::out);
            if(file_3.is_open()){
                cout<<F<<" is opened"<<endl;
            }
            else{
                cout<<F<<" not opened"<<endl;
            }
            //OPEN (3,FILE=F)
        }
    }
    //dout
    //memget(VAR,0.e00,sizeof(*VAR));
    for(int i=0;i<24;i++){
        for(int j=0;j<NCELLS+1;j++)
            get (VAR , i,j)=0.e00;
    }
    if(IFX==0)
        A=(double)FNUM/(FTIME-TISAMP);
    for(JJ=1;JJ<=2;JJ++)
    {
        if(IFX==1)
            A=FNUM/(2.e00*PI*XB[JJ]*(FTIME-TISAMP));
        if(IFX==2)
            A=FNUM/(4.e00*PI*XB[JJ]*XB[JJ]*(FTIME-TISAMP));
        //--JJ=1 for surface at XB(1), JJ=2 for surface at XB(2)
        if(ITYPE[JJ]==2)
        {
            //dout
            //memget(SUMS,0.e00,sizeof(SUMS));
            for(int i=0;i<10;i++){
                for(int j=0;j<3;j++)
                    get (SUMS,1+ i,j)=0.e00;
            }
            for( L=1;L<=MSP;L++)
            {
                for(J=0;J<=8;J++)
                {
                    for(IJ=1;IJ<=2;IJ++)
                    {
                        get (SUMS,1+ J,IJ)=get (SUMS,1+ J,IJ)+get (CSS , 1+ J,JJ,L,IJ);
                    }
                }
            }
            get (VARS , 1,JJ)=get (SUMS,1+ 0,1);
            get (VARS , 2,JJ)=get (SUMS,1+ 1,1);
            get (VARS , 3,JJ)=get (SUMS,1+ 1,2);
            get (VARS , 4,JJ)=get (SUMS,1+ 1,1)*A;
            get (VARS , 5,JJ)=get (SUMS,1+ 1,2)*A;
            get (VARS , 6,JJ)=get (SUMS,1+ 2,1)*A;
            get (VARS , 7,JJ)=get (SUMS,1+ 2,2)*A;
            get (VARS , 8,JJ)=get (SUMS,1+ 3,1)*A;
            get (VARS , 9,JJ)=get (SUMS,1+ 3,2)*A;
            get (VARS , 10,JJ)=get (SUMS,1+ 4,1)*A;
            get (VARS , 11,JJ)=get (SUMS,1+ 4,2)*A;
            get (VARS , 12,JJ)=get (SUMS,1+ 5,1)*A;
            get (VARS , 13,JJ)=get (SUMS,1+ 5,2)*A;
            get (VARS , 14,JJ)=get (SUMS,1+ 6,1)*A;
            get (VARS , 15,JJ)=get (SUMS,1+ 6,2)*A;
            get (VARS , 16,JJ)=get (SUMS,1+ 7,1)*A;
            get (VARS , 17,JJ)=get (SUMS,1+ 7,2)*A;
            get (VARS , 34,JJ)=get (SUMS,1+ 8,1)*A;
            get (VARS , 35,JJ)=get (SUMS,1+ 8,2)*A;
            //   VARS(17,JJ)=SUMS(9,1)*A        //--SURFACE REACTIONS NOT YET IMPLEMENTED
            //   VARS(18,JJ)=SUMS(9,2)*A
            if(get (CSSS , 1,JJ)>1.e-6)
            {
                get (VARS , 20,JJ)=get (CSSS , 3,JJ)/get (CSSS , 2,JJ); ////--n.b. must be modified to include second component in 3D
                get (VARS , 21,JJ)=(get (CSSS , 4,JJ)-get (CSSS , 2,JJ)*get (VARS , 20,JJ)*get (VARS , 20,JJ))/(get (CSSS , 1,JJ)*3.e00*BOLTZ)-TSURF[JJ];
                get (VARS , 20,JJ)=get (VARS , 20,JJ)-VSURF[JJ];
                if(get (CSSS , 6,JJ)>0e00)
                {
                    get (VARS , 22,JJ)=(2.e000/BOLTZ)*(get (CSSS , 5,JJ)/get (CSSS , 6,JJ))-TSURF[JJ];
                }
                else
                {
                    get (VARS , 22,JJ)=0.e00;
                }
            }
            else
            {
                get (VARS ,1+ 19,JJ)=0.e00;
                get (VARS ,1+ 20,JJ)=0.e00;
                get (VARS ,1+ 21,JJ)=0.e00;
            }
            get (VARS ,1+ 22,JJ)=(get (SUMS,1+ 2,1)+get (SUMS,1+ 2,2))*A;
            get (VARS ,1+ 23,JJ)=(get (SUMS,1+ 3,1)+get (SUMS,1+ 3,2))*A;
            get (VARS ,1+ 24,JJ)=(get (SUMS,1+ 4,1)+get (SUMS,1+ 4,2))*A;
            get (VARS ,1+ 25,JJ)=(get (SUMS,1+ 5,1)+get (SUMS,1+ 5,2))*A;
            get (VARS ,1+ 26,JJ)=(get (SUMS,1+ 6,1)+get (SUMS,1+ 6,2))*A;
            get (VARS ,1+ 27,JJ)=(get (SUMS,1+ 7,1)+get (SUMS,1+ 7,2))*A;
            get (VARS ,1+ 28,JJ)=(get (SUMS,1+ 9,1)+get (SUMS,1+ 9,2))*A;
            get (VARS ,1+ 29,JJ)=get (VARS ,1+ 11,JJ)+get (VARS ,1+ 13,JJ)+get (VARS ,1+ 15,JJ)+get (VARS ,1+ 33,JJ);
            get (VARS ,1+ 30,JJ)=get (VARS ,1+ 12,JJ)+get (VARS ,1+ 14,JJ)+get (VARS ,1+ 16,JJ)+get (VARS ,1+ 34,JJ);
            get (VARS ,1+ 31,JJ)=get (VARS ,1+ 29,JJ)+get (VARS ,1+ 30,JJ);
            get (VARS ,1+ 35,JJ)=get (VARS ,1+ 33,JJ)+get (VARS ,1+ 34,JJ);
            for(L=1;MSP;L++)
            {
                if(get (SUMS,1+ 1,1)>0)
                {
                    get (VARS ,1+ 35+L,JJ)=100*get (CSS , 1+ 1,JJ,L,1)/get (SUMS,1+ 1,1);
                }
                else
                {
                    get (VARS ,1+ 35+L,JJ)=0.0;
                }
            }
        }
    }
    //VARSP=0;
    for(int i=0;i<13;i++){
        for(int j=0;j<NCELLS+1;j++){
            for(int k=0;k<MSP+1;k++)
                get (VARSP , i+1,j,k)=0;
        }
    }
    SMCR=0;
    NMCR=0;
    for(N=1;N<=NCELLS;N++)
    {
        if(N==120)
        {
            continue;
        }
        A=FNUM/(get (CELL , 4,N)*NSAMP);
        if(IVB==1)
            A=A*pow((XB[2]-XB[1])/(XB[2]+VELOB*0.5e00*(FTIME+TISAMP)-XB[1]) , IFX+1);
        //--check the above for non-zero XB(1)
        //dout
        //memget(SUM,0,sizeof(SUM));
        for(int i=0;i<14;i++)
            SUM[i]=0;


        
        NMCR+=1;
        //dsuedit
        
        
        //dsuedit
        for(L=1;L<=MSP;L++)
        {
           // dsuedit

            SUM[0]=SUM[0]+get (CS,1+ 0,N,L);
            SUM[1]=SUM[1]+get (CS,1+ 1,N,L);
            SUM[2]=SUM[2]+get (SP , 5,L)*get (CS,1+ 1,N,L);

            
            for(K=1;K<=3;K++)
            {
                SUM[K+2] = SUM[K+2]+get (SP , 5,L)*get (CS,1+ K+1,N,L);
                if(get (CS,1+ 1,N,L)>0.1e00)
                {
                    get (VARSP , K+2,N,L)=get (CS,1+ K+4,N,L)/get (CS,1+ 1,N,L);
                    //--VARSP(2,3,4 are temporarily the mean of the squares of the velocities
                    get (VARSP , K+8+1,N,L)=get (CS,1+ K+1,N,L)/get (CS,1+ 1,N,L);
                }
            }
            SUM[6]=SUM[6]+get (SP , 5,L)*(get (CS,1+ 5,N,L)+get (CS,1+ 6,N,L)+get (CS,1+ 7,N,L));
            SUM[10]=SUM[10]+get (SP , 5,L)*get (CS,1+ 5,N,L);
            SUM[12]=SUM[11]+get (SP , 5,L)*get (CS,1+ 6,N,L);
            SUM[12]=SUM[12]+get (SP , 5,L)*get (CS,1+ 7,N,L);
            SUM[13]=SUM[13]+get (CS,1+ 9,N,L);
            if(get (CS,1+ 1,N,L)>0.5e00)
                SUM[7]=SUM[7]+get (CS,1+ 5,N,L)+get (CS,1+ 6,N,L)+get (CS,1+ 7,N,L);
            if(get(ISPR ,1,L)>0)
            {
                SUM[8]=SUM[8]+get (CS,1+ 8,N,L);
                SUM[9]=SUM[9]+get (CS,1+ 1,N,L)*get(ISPR ,1,L);
            }
        }
        AVW=0;
        for(L=1;L<=MSP;L++)
        {
            get (VARSP , 0+1,N,L)=get (CS,1+ 1,N,L) ;
            get (VARSP , 1+1,N,L)=0.e00;
            get (VARSP , 6+1,N,L)=0.0;
            get (VARSP , 7+1,N,L)=0.0;
            get (VARSP , 8+1,N,L)=0.0;
            if(SUM[1]>0.1)
            {
                get (VARSP , 1+1,N,L)=get (CS,1+ 1,N,L)/SUM[1];
                AVW=AVW+get (SP , 3,L)*get (CS,1+ 1,N,L)/SUM[1];
                if(get(ISPR ,1,L)>0 && get (CS,1+ 1,N,L)>0.5)
                    get (VARSP , 6+1,N,L)=(2.e00/BOLTZ)*get (CS,1+ 8,N,L)/((double)(get(ISPR ,1,L))*get (CS,1+ 1,N,L));
            }
            get (VARSP , 5+1,N,L)=0;
            for(K=1;K<=3;K++)
            {
                get (VARSP , K+1+1,N,L)=(get (SP , 5,L)/BOLTZ)*(get (VARSP , K+1+1,N,L)-pow(get (VARSP , K+8+1,N,L),2));
                get (VARSP , 5+1,N,L)=get (VARSP , 5+1,N,L)+get (VARSP , K+1+1,N,L);
            }
            get (VARSP , 5+1,N,L)=get (VARSP , 5+1,N,L)/3.e00;
            get (VARSP , 8+1,N,L)=(3.e00*get (VARSP , 5+1,N,L)+(double)get(ISPR ,1,L)*get (VARSP , 6+1,N,L))/(3.e00+(double)(get(ISPR ,1,L)));
        }
        if(IVB==0)
            get (VAR , 1,N)=get (CELL , 1,N);
        if(IVB==1)
        {
            C=(XB[2]+VELOB*FTIME-XB[1])/(double)(NDIV); //new DDIV
            get (VAR , 1,N)=XB[1]+((double)(N-1)+0.5)*C;
        }
        get (VAR , 2,N)=SUM[0];

        for(int II = 0 ; II <11 ; II++)
        {
            cout << "SUM["<<II<<"]=" <<SUM[II] <<endl ;
        }

        if(SUM[1]>0.5)
        {
            get (VAR , 3,N)=SUM[1]*A; //--number density Eqn. (4.28)
            get (VAR , 4,N)=get (VAR , 3,N)*SUM[2]/SUM[1]; //--density  Eqn. (4.29)
            get (VAR , 5,N)=SUM[3]/SUM[2];//--u velocity component  Eqn. (4.30)
            get (VAR , 6,N)=SUM[4]/SUM[2]; //--v velocity component  Eqn. (4.30)
            get (VAR , 7,N)=SUM[5]/SUM[2]; //--w velocity component  Eqn. (4.30)
            UU= pow(get (VAR , 5,N),2)+pow(get (VAR , 6,N),2)+pow(get (VAR , 7,N),2);
            if(SUM[1]>1)
            {   
                get (VAR , 8,N)=(fabs(SUM[6]-SUM[2]*UU))/(3.e00*BOLTZ*SUM[1]); //Eqn. (4.39)
                //--translational temperature
                get (VAR , 19,N)=fabs(SUM[10]-SUM[2]*pow(get (VAR , 5,N),2))/(BOLTZ*SUM[1]);
                get (VAR , 20,N)=fabs(SUM[11]-SUM[2]*pow(get (VAR , 6,N),2))/(BOLTZ*SUM[1]);
                get (VAR , 21,N)=fabs(SUM[12]-SUM[2]*pow(get (VAR , 7,N),2))/(BOLTZ*SUM[1]);
            }
            else
            {
                get (VAR , 8,N)=1.0;
                get (VAR , 19,N)=1.0;
                get (VAR , 20,N)=1.0;
                get (VAR , 21,N)=1.0;
            }
            if(SUM[9]>0.1e00)
            {
                get (VAR , 9,N)=(2.e00/BOLTZ)*SUM[8]/SUM[9]; ////--rotational temperature Eqn. (4.36)
            }
            else
                get (VAR , 9,N)=0.0;
            
            get (VAR , 10,N)=FTMP[1]; ////vibration default
            DOF=(3.e00+SUM[9]/SUM[1]);
            get (VAR , 11,N)=(3.0*get (VAR , 8,N)+(SUM[9]/SUM[1])*get (VAR , 9,N))/DOF;
            //--overall temperature based on translation and rotation
            get (VAR , 18,N)=get (VAR , 3,N)*BOLTZ*get (VAR , 8,N);
            //--scalar pressure (now (from V3) based on the translational temperature)
            if(MMVM>0)
            {
                for(L=1;L<=MSP;L++)
                {
                    VDOF[L]=0.0;
                    //dout
                    if(get(ISPV  ,L) > 0)
                    {
                        for(K=1;K<=get(ISPV  ,L);K++)
                        {
                            if(get (CS,1+ K+9,N,L)<BOLTZ)
                            {
                                get(TV , K,L)=0.0;
                                get(DF , N,K,L)=0.0;
                            }
                            else
                            {
                                get(TV , K,L)=get(SPVM ,1,K,L)/log(1.0+get (CS,1+ 1,N,L)/get (CS,1+ K+9,N,L)) ;//--Eqn.(4.45)
                                get(DF , N,K,L)=2.0*(get (CS,1+ K+9,N,L)/get (CS,1+ 1,N,L))*log(1.0+get (CS,1+ 1,N,L)/get (CS,1+ K+9,N,L)); //--Eqn. (4.46)
                            }
                            VDOF[L]=VDOF[L]+get(DF , N,K,L);
                        }
                        //memget(TVIB,0.0,sizeof(*TVIB));
                        for(int i=0;i<MSP+1;i++)
                            TVIB[i]=0.0;
                        
                        for(K=1;K<=get(ISPV  ,L);K++)
                        {
                            if(VDOF[L]>1.e-6)
                            {
                                TVIB[L]=TVIB[L]+get(TV , K,L)*get(DF , N,K,L)/VDOF[L] ;
                            }
                            else
                                TVIB[L]=FVTMP[1] ;
                        }
                    }
                    else
                    {
                        TVIB[L]=TREF;
                        VDOF[L]=0.0;
                    }
                    get (VARSP , 7+1,N,L)=TVIB[L];
                }
                VDOFM=0.0;
                TVIBM=0.0;
                A=0.e00;
                for(L=1;L<=MSP;L++)
                {
                    //doubt
                    if(get(ISPV  ,L) > 0)
                    {
                        A=A+get (CS,1+ 1,N,L);
                    }
                }
                for(L=1;L<=MSP;L++)
                {
                    //dout
                    if(get(ISPV  ,L) > 0)
                    {
                        VDOFM=VDOFM+VDOF[L]*get (CS,1+ 1,N,L)/A;
                        TVIBM=TVIBM+TVIB[L]*get (CS,1+ 1,N,L)/A;
                    }
                }
                get (VAR , 10,N)=TVIBM;
            }
            for(L=1;L<=MSP;L++)
            {
                if(get (VARSP , 0+1,N,L)>0.5)
                {
                    //--convert the species velocity components to diffusion velocities
                    for(K=1;K<=3;K++)
                    {
                        get (VARSP , K+8+1,N,L)=get (VARSP , K+8+1,N,L)-get (VAR , K+4,N);
                    }
                    if(MELE>1)
                    {
                        //--calculate the electronic temperatures for the species
                        //memget(ELDOF,0.e00,sizeof(*ELDOF));
                        for(int i=0;i<MSP+1;i++)
                            ELDOF[i] = 0.e00;
                        //dout
                        //memget(TEL,0.e00,sizeof(*TEL));
                        for(int i=0;i<MSP+1;i++)
                            TEL[i] = 0.e00;
                        if(MELE>1)
                        {
                            A=0.e00;
                            B=0.e00;
                            for(M=1;M<=get(NELL  ,L);M++)
                            {
                                if(get (VARSP , 5+1,N,L)>1.e00)
                                {
                                    C=get(QELC ,2,M,L)/get (VARSP , 5+1,N,L);
                                    A=A+get(QELC ,1,M,L)*exp(-C);
                                    B=B+get(QELC ,1,M,L)*C*exp(-C);
                                }
                            }
                            if(B>1.e-10)
                            {
                                TEL[L]=(get (CS,1+ 9,N,L)/get (CS,1+ 1,N,L))/(BOLTZ*B/A);
                            }
                            else
                                TEL[L]=get (VAR , 11,N);
                            get (VARSP , 12+1,N,L)=TEL[L];
                            ELDOF[L]=0.e00;
                            if(get (VARSP , 5+1,N,L)>1.e00)
                                ELDOF[L]=2.e00*(get (CS,1+ 9,N,L)/get (CS,1+ 1,N,L))/(BOLTZ*get (VARSP , 5+1,N,L) ) ;
                            if(ELDOF[L]<0.01)
                            {
                                get (VARSP , 12+1,N,L)=get (VAR , 11,N);
                            }
                        }
                        else
                        {
                            ELDOF[L]=0.0;
                        }
                    }
                }
                else
                {
                    for(K=8;K<=12;K++)
                    {
                        get (VARSP , K+1,N,L)=0.e00;
                    }
                }
            }
            //--set the overall electronic temperature
            if(MELE>1)
            {
                C=0.e00;
                for(L=1;L<=MSP;L++)
                {
                    if(ELDOF[L]>1.e-5)
                        C=C+get (CS,1+ 1,N,L);
                }
                if(C>0.e00)
                {
                    A=0.e00;
                    B=0.e00;
                    for(L=1;L<=MSP;L++)
                    {
                        if(ELDOF[L]>1.e-5)
                        {
                            A=A+get (VARSP , 12+1,N,L)*get (CS,1+ 1,N,L);
                            B=B+get (CS,1+ 1,N,L);
                        }
                    }
                    get (VAR , 22,N)=A/B;
                }
                else{
                    get (VAR , 22,N)=get (VAR , 11,N);
                }
            }
            else{
                get (VAR , 22,N)=FTMP[1] ;
            }
            if(MMVM>0)
            {
                //--set the overall temperature and degrees of freedom for the individual species
                for(L=1;L<=MSP;L++)
                {
                    if(MELE>1){
                        SDOF[L]=3.e00+get(ISPR ,1,L)+VDOF[L]+ELDOF[L];
                        get (VARSP , 8+1,N,L)=(3.0*get (VARSP , 5+1,N,L)+get(ISPR ,1,L)*get (VARSP , 6+1,N,L)+VDOF[L]*get (VARSP , 7+1,N,L)+ELDOF[L]*get (VARSP , 12+1,N,L))/SDOF[L];
                    }
                    else{
                        SDOF[L]=3.e00+get(ISPR ,1,L)+VDOF[L];
                        get (VARSP , 8+1,N,L)=(3.0*get (VARSP , 5+1,N,L)+get(ISPR ,1,L)*get (VARSP , 6+1,N,L)+VDOF[L]*get (VARSP , 7+1,N,L))/SDOF[L];
                    }
                }
                //--the overall species temperature now includes vibrational and electronic excitation
                //--the overall gas temperature can now be set
                A=0.e00;
                B=0.e00;
                for(L=1;L<=MSP;L++)
                {
                    A=A+SDOF[L]*get (VARSP , 8+1,N,L)*get (CS,1+ 1,N,L);
                    B=B+SDOF[L]*get (CS,1+ 1,N,L);
                }
                get (VAR , 11,N)=A/B ;
            }
            VEL=sqrt(pow(get (VAR , 5,N),2)+pow(get (VAR , 6,N),2)+pow(get (VAR , 7,N),2));
            get (VAR , 12,N)=VEL/sqrt((DOF+2.e00)*get (VAR , 11,N)*(SUM[1]*BOLTZ/SUM[2])/DOF);
            //--Mach number
            get (VAR , 13,N)=SUM[0]/NSAMP; ////--average number of molecules in cell
            //dout
            if(COLLS[N] > 2.0)
            {
                get (VAR , 14,N)=0.5e00*(FTIME-TISAMP)*(SUM[1]/NSAMP)/WCOLLS[N];
                //--mean collision time
                get (VAR , 15,N)=0.92132e00*sqrt(fabs(SUM[7]/SUM[1]-UU))*get (VAR , 14,N);
                //--mean free path (based on r.m.s speed with correction factor based on equilibrium)
                get (VAR , 16,N)=CLSEP[N]/(COLLS[N]*get (VAR , 15,N));
            }
            else{
                get (VAR , 14,N)=1.e10;
                get (VAR , 15,N)=1.e10/get (VAR , 3,N);
                //--m.f.p set by nominal values
            }
        }
        else
        {
            for(L=3;L<=22;L++)
            {
                get (VAR , L,N)=0.0;
            }
        }
        get (VAR , 17,N)=VEL;
    }
    if(FTIME>0.5e00*DTM)
    {
        if(ICLASS==1){
            if(IFX==0)
                file_3<<"DSMC program for a one-dimensional plane flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
            if(IFX==1)
                file_3<<"DSMC program for a cylindrical flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
            if(IFX==2)
                file_3<<"DSMC program for a spherical flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
        }
        file_3<<endl;//WRITE (3,*)
        file_3<<"Interval "<<NOUT<<" Time "<<FTIME<< " with "<<NSAMP<<" samples from "<<TISAMP<<endl;
        //WRITE (3,*) 'Interval',NOUT,'Time ',FTIME, ' with',NSAMP,' samples from',TISAMP
        //990 FORMAT(I7,G13.5,I7,G13.5)
        //Dout
        NNN=TOTMOV;
        cout<<"TOTAL MOLECULES = "<< NM<<endl;
        //dout
        //NMS=0;
        for(int i=0;i<MSP+1;i++)
            NMS[i]=0;

        for(N=1;N<=NM;N++)
        {
            M=get(IPSP ,N);
            NMS[M]+=1;
        }
        file_3<<"Total simulated molecules = "<<NM<<endl;
        for(N=1;N<=MSP;N++)
        {
            cout<< " SPECIES "<<N<<" TOTAL = "<<NMS[N]<<endl;
            file_3<<"Species "<<N<<" total = "<<NMS[N]<<endl;
        }
        if(MEX>0)
        {
            ENERGY(0,A);
            for(N=1;N<=MSP;N++)
            {
                if(get(ISPV  ,N)>0){
                    file_9<< "SP "<<N<<" DISSOCS "<<TDISS[N]<<" RECOMBS "<<TRECOMB[N]<<endl;
                    cout<<"SP"<<N<<"DISSOCS"<<TDISS[N]<<" RECOMBS "<<TRECOMB[N]<<endl;
                    file_3<<"SP "<<N<<" DISSOCS "<<TDISS[N]<<" RECOMBS "<<TRECOMB[N]<<endl;
                }
            }
            for(N=1;N<=MEX;N++)
            {
                cout<<"EX,C reaction\t"<<N<<" number"<<TNEX[N]<<endl;
                file_9<<"EX,C reaction\t "<<N<<" number "<<TNEX[N]<<endl;
                file_3<<"EX,C reaction \t"<<N<<" number "<<TNEX[N]<<endl;
                
            }
        }
        
        file_3<<"Total molecule moves   = "<<NNN<<endl;
        //dout
        NNN=TOTCOL;
        file_3<<"Total collision events = "<<NNN<<endl;
        //
        file_3<<"Species dependent collision numbers in current sample"<<endl;
        for(N=1;N<=MSP;N++)
        {
            if(IGAS!=8){
                for(M=1;M<=MSP;M++)
                    file_3<<get (TCOL , N,M)<<"\t";
                file_3<<endl;
                //WRITE(3,901) (get (TCOL , N,M),M=1,MSP);
            }
            if(IGAS==8){
                for(M=1;M<=MSP;M++)
                    file_3<<get (TCOL , N,M)<<"\t";
                file_3<<endl;
                // WRITE(3,902) (get (TCOL , N,M),M=1,MSP);
            }
        }
        //Dout
        //901 FORMAT(5G13.5)
        //902 FORMAT(8G13.5)
        //dout
        CTIME=clock();
        file_3<<"Computation time "<<(double)CTIME/1000.0<< "seconds"<<endl;
        file_3<<"Collision events per second "<<(TOTCOL-TOTCOLI)*1000.e00/(double)CTIME<<endl;
        file_3<<"Molecule moves per secon "<<(TOTMOV-TOTMOVI)*1000.e00/(double)CTIME<<endl;
        if(ICLASS==0 && MMVM==0 && ISF==0){
            //--a homogeneous gas with no vibratioal modes - assume that it is a collision test run
            //*PRODUCES DATA FOR TABLES 6.1 AND 6.2 IN SECTION 6.2*
            //
            A=0.e00;
            B=0.e00;
            C=0.e00;
            for(N=1;N<=NCCELLS;N++)
            {
                A+=get (CCELL , 5,N);
                B+=get (CCELL , 4,N);
                C+=get (CCELL , 3,N);
            }
            file_3<<"Overall time step "<<DTM<<endl;
            file_3<<"Molecules per collision cell "<<(double)(NM)/(double)(NCCELLS)<<endl;
            file_3<<"Mean cell time ratio "<< A/((double)(NCCELLS)*FTIME)<<endl;
            file_3<<"Mean value of cross-section and relative speed "<<B/(double)(NCCELLS)<<endl;
            file_3<<"Mean half collision cell time step "<<C/(double)(NCCELLS)<<endl;
            if(MSP==1){
                A=2.e00*SPI*get (VAR , 3,1)  *(pow(get (SP , 1,1),2))*sqrt(4.e00*BOLTZ*get (SP , 2,1)/get (SP , 5,1))*pow((get (VAR , 11,1))/get (SP , 2,1),(1.e00-get (SP , 3,1)));
                //--Eqn. (2.33) for equilibhrium collision rate
                file_3<<"Coll. rate ratio to equilib "<<get (TCOL , 1,1)/((double)(NM)*(FTIME-TISAMP))/A<<endl;
            }
            else{
                file_3<<"Species collision rate ratios to equilibrium"<<endl;
                for(N=1;N<=MSP;N++){
                    file_3<<"Collision rate for species "<<N<<endl;
                    for(M=1;M<=MSP;M++)
                    {
                        THCOL[N,M]=2.e00*(1.e00/SPI)*get (VAR , 3,1)*get (VARSP , 1+1,1,M)*get (SPM , 2,N,M)*sqrt(2.e00*BOLTZ*get (SPM , 5,N,M)/get (SPM , 1,N,M))*pow(get (VAR , 11,1)/get (SPM , 5,N,M),1.e00-get (SPM , 3,N,M));
                        //--Eqn. (2.36) for equilibhrium collision rate of species N with species M
                        file_3<<" with species "<<M<<" is "<<get (TCOL , N,M)/((double)(NM)*get(FSP ,N,1)*(FTIME-TISAMP))/THCOL[N,M]<<endl;
                    }
                }
                file_3<<endl;
                for(N=1;N<=MSP;N++){
                    file_3<<"Collision numbers for species "<<N<<endl;
                    for(M=1;M<=MSP;M++){
                        file_3<<"with species "<<M<<" "<<get (TCOL , N,M)<<endl;
                    }
                }
            }
        }
        file_3<<endl;
        if(ITYPE[1]==2|| ITYPE[2]==2)
            file_3<<"Surface quantities"<<endl;
        for(JJ=1;JJ<=2;JJ++)
        {
            if(ITYPE[JJ]==2){
                file_3<<endl;
                file_3<<"Surface at "<<XB[JJ]<<endl;
                file_3<<"Incident sample "<<get (VARS ,1+ 0,JJ)<<endl;
                file_3<<"Number flux "<<get (VARS ,1+ 3,JJ)<<" /sq m/s"<<endl;
                file_3<<"Inc pressure "<<get (VARS ,1+ 5,JJ)<<" Refl pressure "<<get (VARS ,1+ 6,JJ)<<endl;
                file_3<<"Pressure "<< get (VARS ,1+ 5,JJ)+get (VARS ,1+ 6,JJ)<<" N/sq m"<<endl;
                file_3<<"Inc y shear "<<get (VARS ,1+ 7,JJ)<<" Refl y shear "<<get (VARS ,1+ 8,JJ)<<endl;
                file_3<<"Net y shear "<<get (VARS ,1+ 7,JJ)-get (VARS ,1+ 8,JJ)<<" N/sq m"<<endl;
                file_3<<"Net z shear "<<get (VARS ,1+ 9,JJ)-get (VARS ,1+ 10,JJ)<<" N/sq m"<<endl;
                file_3<<"Incident translational heat flux "<<get (VARS ,1+ 11,JJ)<<" W/sq m"<<endl;
                if(MMRM>0)
                    file_3<<"Incident rotational heat flux "<<get (VARS ,1+ 13,JJ)<<" W/sq m"<<endl;
                if(MMVM>0)
                    file_3<<"Incident vibrational heat flux "<<get (VARS ,1+ 15,JJ)<<" W/sq m"<<endl;
                if(MELE>1)
                    file_3<<"Incident electronic heat flux "<<get (VARS ,1+ 33,JJ)<<" W/sq m"<<endl;
                file_3<<"Total incident heat flux "<<get (VARS ,1+ 29,JJ)<<" W/sq m"<<endl;
                file_3<<"Reflected translational heat flux "<<get (VARS ,1+ 12,JJ)<<" W/sq m"<<endl;
                if(MMRM>0)
                    file_3<<"Reflected rotational heat flux "<<get (VARS ,1+ 14,JJ)<<" W/sq m"<<endl;
                if(MMVM>0)
                    file_3<<"Reflected vibrational heat flux "<<get (VARS ,1+ 16,JJ)<<" W/sq m"<<endl;
                if(MELE>1)
                    file_3<<"Reflected electronic heat flux "<<get (VARS ,1+ 34,JJ)<<" W/sq m"<<endl;
                file_3<<"Total reflected heat flux "<<get (VARS ,1+ 30,JJ)<<" W/sq m"<<endl;
                file_3<<"Net heat flux "<<get (VARS ,1+ 31,JJ)<<" W/sq m"<<endl;
                file_3<<"Slip velocity (y direction) "<<get (VARS ,1+ 19,JJ)<<" m/s"<<endl;
                file_3<<"Translational temperature slip"<<get (VARS ,1+ 20,JJ)<<" K"<<endl;
                if(MMRM>0)
                    file_3<<"Rotational temperature slip "<<get (VARS ,1+ 21,JJ)<<" K"<<endl;
                if(MSP>1)
                {
                    for(L=1;L<=MSP;L++)
                    {
                        file_3<<"Species "<<L<<" percentage "<<get (VARS ,1+ L+35,JJ)<<endl;
                    }
                }
            }
        }

        file_3<<endl;
        //PPA=0;
        for(int i=0;i<MSP+1;i++)
            PPA[i]=0;

        for(N=1;N<=NCELLS;N++)
        {
            for(M=1;M<=MSP;M++){
                PPA[M]=PPA[M]+get (VARSP , 1,N,M);
            }
        }
        // WRITE (*,*)
        //cin.get();
        if(MSP>1)
        {
            file_3<<"GAINS FROM REACTIONS"<<endl;
            file_3<<"                          Dissoc.     Recomb. Endo. Exch.  Exo. Exch."<<endl;
            for(M=1;M<=MSP;M++){
                file_3<<"                          SPECIES "<<M<<" "<<get (TREACG , 1,M)<<" "<<get (TREACG , 2,M)<<" "<<get (TREACG , 3,M)<<" "<<get (TREACG , 4,M)<<endl;
            }
            file_3<<endl;
            file_3<<"LOSSES FROM REACTIONS"<<endl;
            file_3<<"                          Dissoc.     Recomb. Endo. Exch.  Exo. Exch."<<endl;
            for(M=1;M<=MSP;M++){
                file_3<<"                          SPECIES "<<M<<" "<<get (TREACL , 1,M)<<" "<<get (TREACL , 2,M)<<" "<<get (TREACL , 3,M)<<" "<<get (TREACL , 4,M)<<endl;
            }
            file_3<<endl;
            file_3<<"TOTALS"<<endl;
            for(M=1;M<=MSP;M++){
                file_3<<"                        SPECIES "<<M<<" GAINS "<<get (TREACG , 1,M)+get (TREACG , 2,M)+get (TREACG , 3,M)+get (TREACG , 4,M)<<" LOSSES "<<get (TREACL , 1,M)+get (TREACL , 2,M)+get (TREACL , 3,M)+get (TREACL , 4,M)<<endl;
            }
        }
        file_3<<endl;
        file_3<<"Flowfield properties "<<endl;
        file_3<< NSAMP<<" Samples"<<endl;
        file_3<<"Overall gas"<<endl;
        cout << "NCELLS = " << NCELLS << endl ;
        file_3<<"Cell\tx coord.\tSample\tNumber Dens.\t Density\tu velocity\tv velocity\tw velocity\tTrans. Temp.\tRot. Temp.\tVib. Temp. \tEl. Temp. \tTemperature \tMach no. \tMols/cell\tm.c.t   \tm.f.p\tmcs/mfp\tspeed \tPressure \tTTX \tTTY\tTTZ\tSpecies Fractions "<<endl;
        for(N=1;N<=NCELLS;N++)
        {
            file_3<< N<<" \t";
            for(M=1;M<=10;M++){
                file_3<<get (VAR , M,N)<<"\t";
            }
            file_3<<get (VAR , 22,N)<<"\t ";
            for(M=11;M<=21;M++){
                file_3<<get (VAR , M,N)<<" \t";
            }
            for(L=1;M<=MSP;M++){
                file_3<<get (VARSP , 2,N,L)<<"\t ";
            }
            file_3<<endl;
        }

        cout <<" sum[2] = "<< SUM[2] << endl ; // dsuedit
        cout << " sum[3] = " << SUM[3] << endl ; // dsuedit
        file_3<<"Individual molecular species"<<endl;
        for(L=1;L<=MSP;L++){
            file_3<<"Species "<<L<<endl;
            file_3<<"Cell\t x coord.  \t    Sample  \t     Percentage \t  Species TTx  \t Species TTy \t Species TTz \t Trans. Temp.\t  Rot. Temp.\t  Vib. Temp. \t  Spec. Temp  \tu Diff. Vel.\t v Diff. Vel.\t w. Diff. Vel.\t Elec. Temp."<<endl;
            for(N=1;N<=NCELLS;N++){
                file_3<< N<<" "<<get (VAR , 1,N)<<" \t";
                for(M=0;M<=12;M++)
                    file_3<<get (VARSP , M+1,N,L)<<"\t ";
                file_3<<endl;
            }
        }
        //dout
        //999 FORMAT (I5,30G13.5)
        //998 FORMAT (G280.0)
        // 997 FORMAT (G188.0)
        // CLOSE (3)
        file_3.close();
    }
    if(ICLASS==0 && ISF==1){
        //--a homogeneous gas and the "unsteady sampling" option has been chosen-ASSUME THAT IT IS A RELAXATION TEST CASE FOR SECTION 6.2
        INITIALISE_SAMPLES();
        //write a special output file for internal temperatures and temperature versus collision number
        //dout
        file_10.open("RELAX.DAT", ios::app | ios::out);
        if(file_10.is_open()){
            cout<<"RELAX.DAT is opened"<<endl;
        }
        else{
            cout<<"RELAX.DAT not opened"<<endl;
        }
        // OPEN (10,FILE='RELAX.DAT',ACCESS='APPEND')
        A=2.0*TOTCOL/NM; //--mean collisions
        //--VAR(11,N)   //--overall
        //--VAR(8,N)    //--translational
        //--VAR(9,N)    //--rotational
        //--VAR(10,N)   //--vibrational
        //--VAR(22,N)   //--electronic
        //file_10<<std::right<<setw(15)<<A<<setw(15)<<get (VAR , 8,1)<<setw(15)<<get (VAR , 9,1)<<setw(15)<<get (VAR , 8,1)-get (VAR , 9,1)<<endl;
        file_10<<std::right<<setw(15)<<A<<setw(15)<<get (VAR , 11,1)<<setw(15)<<get (VAR , 8,1)<<setw(15)<<get (VAR , 9,1)<<setw(15)<<get (VAR , 10,1)<<setw(15)<<get (VAR , 22,1)<<endl;
        //file_10<<std::right<<setw(15)<<A<<setw(15)<<get (VAR , 8,1]<<setw(15)<<get (VAR , 9,1]<<setw(15)<<get (VAR , 8,1]-get (VAR , 9,1]<<endl;
        //  WRITE (10,950) A,VAR(8,1),VAR(9,1),VAR(8,1)-VAR(9,1)   //--Generates output for Figs. 6.1 and 6.2
        //  WRITE (10,950) A,VAR(11,1),VAR(8,1),VAR(9,1),VAR(10,1),VAR(22,1)   //--Generates output for modal temperatures in Figs. 6.3, 6.5 +
        //  WRITE (10,950) A,0.5D00*(VAR(8,1)+VAR(9,1)),VAR(10,1),0.5D00*(VAR(8,1)+VAR(9,1))-VAR(10,1)  //--Generates output for Figs. 6.4
        //
        //--VARSP(8,N,L) //--overall temperature of species L
        //  WRITE (10,950) A,VARSP(8,1,3),VARSP(8,1,2),VARSP(8,1,5),VARSP(8,1,4),A  //--output for Fig 6.17
        // CLOSE (10)
        file_10.close();
    }
    //dout
    // 950 FORMAT (6G13.5)
    if(IGAS==8||IGAS==6||IGAS==4)
    {
        //--Write a special output file for the composition of a reacting gas as a function of time
        //dout
        //OPEN (10,FILE='COMPOSITION.DAT',ACCESS='APPEND')
        file_10.open("COMPOSITION.DAT", ios::app | ios::out);
        if(file_10.is_open()){
            cout<<"COMPOSITION.DAT is opened"<<endl;
        }
        else{
            cout<<"COMPOSITION.DAT not opened"<<endl;
        }
        AS=NM;
        //dout
        AT=FTIME*1.e6;
        if (IGAS == 4)
            file_10<< AT <<" "<<(double)(NMS[1])/1000000<<" "<<A<<" "<<get (VAR , 11,1)<<endl;    //--Data for fig
        if (IGAS == 8)
            file_10<<AT<<" "<<NMS[1]/AS<<" "<<NMS[2]/AS<<" "<<NMS[3]/AS<<" "<<NMS[4]/AS<<" "<<NMS[5]/AS<<" "<<NMS[6]/AS<<" "<<NMS[7]/AS<<" "<<NMS[8]/AS<<" "<<get (VAR , 11,1)<<endl;
        if (IGAS == 6)
            file_10<<AT<<" "<<NMS[1]/AS<<" "<<NMS[2]/AS<<" "<<NMS[3]/AS<<" "<<NMS[4]/AS<<" "<<NMS[5]/AS<<" "<<get (VAR , 11,1)<<endl;
        //dout
        // 888 FORMAT(10G13.5)
        file_10.close();
    }
    if(FTIME>0.5e00*DTM){
        //
        //--reset collision and transit times etc.
        //
        cout<<"Output files written "<<endl;
        DTMI=DTM;
        if(IMTS<2){
            if(ICLASS>0)
                DTM*=2;
            //--this makes it possible for DTM to increase, it will be reduced as necessary
            for(NN=1;NN<=NCELLS;NN++)
            {
                CDTM[NN]=DTM;
                B=get (CELL , 3,NN)-get (CELL , 2,NN) ;//--sampling cell width
                if(get (VAR , 13,NN)>20.e00){
                    //consider the local collision rate
                    CDTM[NN]=get (VAR , 14,NN)*CPDTM;
                    //look also at sampling cell transit time based on the local flow speed
                    A=(B/(fabs(get (VAR , 5,NN))))*TPDTM;
                    if(A<CDTM[NN])
                        CDTM[NN]=A;
                }
                else{
                    //-- base the time step on a sampling cell transit time at the refence vmp
                    A=TPDTM*B/VMPM;
                    if(A<CDTM[NN])
                        CDTM[NN]=A;
                }
                if(CDTM[NN]<DTM)
                    DTM=CDTM[NN];
            }
        }
        else
        {
            //dout
            //memget(CDTM, DTM, sizeof(*CDTM));
            for(int i=0;i<NCELLS+1;i++)
                CDTM[i]= DTM;
            //CDTM=DTM;
        }
        for(N=1;N<=NCELLS;N++){
            NN=get (ICCELL , 3,N);
            get (CCELL , 3,N)=0.5*CDTM[NN];
        }
        file_9<<"DTM changes  from "<<DTMI<<" to "<<DTM<<endl;
        DTSAMP=DTSAMP*DTM/DTMI;
        DTOUT=DTOUT*DTM/DTMI;
    }
    else
    {
        INITIALISE_SAMPLES();
    }
    if(ICLASS==1&& ISF==1)
    {
        //****
        //--write TECPLOT data files for x-t diagram (unsteady calculation only)
        //--comment out if not needed
        //dout
        file_18.open("DS1xt.DAT", ios::app | ios::out);
        if(file_18.is_open()){
            cout<<"DS1xt.DAT is opened"<<endl;
        }
        else
            cout<<"DS1xt.DAT not opened"<<endl;
        // OPEN (18,FILE='DS1xt.DAT',ACCESS='APPEND')
        //--make sure that it is empty at the stary of the run
        //SETXT();
        // CLOSE (18)
        file_18.close();
        //****
    }
    //WRITE (19,*) FTIME,-get (VARS ,1+ 5,1],-get (VARS ,1+ 5,1]-get (VARS , 6,1]
    
    file_7.open("PROFILE.DAT" , ios::out);
    if(file_7.is_open()){
        cout<<"PROFILE.DAT is opened"<<endl;
    }
    else
        cout<<"PROFILE.DAT not opened"<<endl;
    // OPEN (7,FILE='PROFILE.DAT',FORM='FORMATTED')
    //
    //OPEN (8,FILE='ENERGYPROF.DAT',FORM='FORMATTED')
    //
    // 995 FORMAT (22G13.5)
    // 996 FORMAT (12G14.6)
    for(N=1;N<=NCELLS;N++)
    {
        //
        //--the following line is the default output
        //  WRITE (7,995) VAR(1,N),VAR(4,N),VAR(3,N),VAR(11,N),VAR(18,N),VAR(5,N),VAR(12,N),VAR(8,N),VAR(9,N),VAR(10,N),VAR(22,N),     &
        //        (VARSP(8,N,M),M=1,MSP),(VARSP(1,N,M),M=1,MSP)
        //
        //--calculate energies per unit mass (employed for re-entry shock wave in Section 7.5)
        C1=0.5e00*pow(get (VAR , 5,N),2);    //--Kinetic
        C2=0.e00;                 //--Thermal
        C3=0.e00;                //--Rotational
        C4=0.e00;               //--Vibrational
        C5=0.e00;              //--Electronic
        C6=0.e00;             //--Formation
        for(L=1;L<=MSP;L++)
        {
            //    C2=C2+3.D00*BOLTZ*VARSP(5,N,L)*VARSP(1,N,L)/SP(5,L)
            A=(get (CS,1+ 1,N,L)/get (VARSP , 2,N,L))*get (SP , 5,L);
            if(get (CS,1+ 1,N,L)>0.5e00){
                C2=C2+0.5e00*(get (CS,1+ 5,N,L)+get (CS,1+ 6,N,L)+get (CS,1+ 7,N,L))*get (SP , 5,L)/A;
                if(get(ISPR ,1,L)>0)
                    C3=C3+get (CS,1+ 8,N,L)/A;
                if(get(ISPV  ,L)>0)
                    C4=C4+get (CS,1+ 10,N,L)*BOLTZ*get(SPVM ,1,1,L)/A;
                if(get(NELL  ,L)>1)
                    C5=C5+get (CS,1+ 9,N,L)/A;
                C6=C6+get (SP , 6,L)*get (CS,1+ 1,N,L)/A;
            }
        }
        C2=C2-C1;
        //  A=0.5D00*VFX(1)*2+2.5D00*BOLTZ*FTMP(1)/(0.75*SP(5,2)+0.25*SP(5,1))
        C7=C1+C2+C3+C4+C5+C6;
        //
        //  WRITE (8,995) VAR(1,N),C1/A,C2/A,C3/A,C4/A,C5/A,C6/A,C7/A
        //
        //--the following lines are for normalised shock wave output in a simple gas (Sec 7.3)
        C1=FND[2]-FND[1];
        C2=FTMP[2]-FTMP[1];
        
        file_7<<get (VAR , 1,N)<<" "<<get (VAR , 2,N)<<" "<<(0.5*(get (VAR , 20,N)+get (VAR , 21,N))-FTMP[1])/C2<<" "<<(get (VAR , 19,N)-FTMP[1])/C2<<" "<<(get (VAR , 11,N)-FTMP[1])/C2<<" "<<(get (VAR , 3,N)-FND[1])/C1<<endl;
        //--the following replaces sample size with density
        //C3=0.D00
        //DO L=1,MSP
        //  C3=C3+FND(1)*FSP(L,1)*SP(5,L)  //--upstream density
        //END DO
        //C4=0.D00
        //DO L=1,MSP
        //  C4=C4+FND(2)*FSP(L,2)*SP(5,L)  //--upstream density
        //END DO
        //
        //  WRITE (7,996) VAR(1,N),(VAR(4,N)-C3)/(C4-C3),(0.5*(VAR(20,N)+VAR(21,N))-FTMP(1))/C2,(VAR(19,N)-FTMP(1))/C2,(VAR(11,N)-FTMP(1))/C2,    &
        //        (VAR(3,N)-FND(1))/C1
        //--the following lines is for a single species in a gas mixture
        //  C1=C1*FSP(3,1)
        //  WRITE (7,996) VAR(1,N),VARSP(1,N,3),(0.5*(VARSP(3,N,3)+VARSP(4,N,3))-FTMP(1))/C2,(VARSP(2,N,3)-FTMP(1))/C2,(VARSP(5,N,3)-FTMP(1))/C2,(VAR(3,N)*VARSP(1,N,3)-FND(1)*FSP(3,1))/C1
        //
        //--the following line is for Couette flow (Sec 7.4)
        //  WRITE (7,996) VAR(1,N),VAR(2,N),VAR(5,N),VAR(6,N),VAR(7,N),VAR(11,N)
        //--the following line is for the breakdown of equilibrium in expansions (Sec 7.10)
        //  WRITE (7,996) VAR(1,N),VAR(2,N),VAR(12,N),VAR(4,N),VAR(5,N),VAR(8,N),VAR(9,N),VAR(10,N),VAR(11,N),VAR(19,N),VAR(20,N),VAR(21,N)
        //
    }
    if(ISF==1)
        INITIALISE_SAMPLES();
    // CLOSE(7)
    file_7.close();
    //
    //--deallocate local variables
    //
    //dout
    
    // DEALLOCATE (TV,TVIB,VDOF,THCOL,STAT=ERROR)
    // if(ERROR)
    //     cout<<"PROGRAM COULD NOT DEALLOCATE OUTPUT VARIABLES"<<ERROR;
    TOUT=TOUT+DTOUT;
   
    return;

}

void cuda_collisions(int N)
{
    //CALC calc;
    //MOLECS molecs;
    //GAS gas;
    //OUTPUT output;
    //GEOM_1D geom;
    int NN,M,MM,L,LL,K,KK,KT,J,I,II,III,NSP,MAXLEV,IV,NSEL,KV,LS,MS,KS,JS,IIII,LZ,KL,IS,IREC,NLOOP,IA,IDISS,IEX,NEL,NAS,NPS,
    JJ,LIMLEV,KVV,KW,INIL,INIM,JI,LV,IVM,NMC,NVM,LSI,JX,MOLA,KR,JKV,NSC,KKV,IAX,NSTEP,NTRY,NLEVEL,NSTATE,IK,NK,MSI ;
    double A,AA,AAA,AB,B,BB,BBB,ABA,ASEL,DTC,SEP,VR,VRR,ECT,EVIB,ECC,ZV,ERM,C,OC,SD,D,CVR,PROB,RML,RMM,ECTOT,ETI,EREC,ET2,
    XMIN,XMAX,WFC,CENI,CENF,VRRT,EA,DEN,E1,E2,VRI,VRA ;
    double VRC[4],VCM[4],VRCP[4],VRCT[4];
    //   //N,M,K working integer
    // //LS,MS,KS,JS molecular species
    // //VRC components of the relative velocity
    // //RML,RMM molecule mass parameters
    // //VCM components of the center of mass velocity
    // //VRCP post-collision components of the relative velocity
    // //SEP the collision partner separation
    // //VRR the square of the relative speed
    // //VR the relative speed
    // //ECT relative translational energy
    // //EVIB vibrational energy
    // //ECC collision energy (rel trans +vib)
    // //MAXLEV maximum vibrational level
    // //ZV vibration collision number
    // //SDF the number of degrees of freedom associated with the collision
    // //ERM rotational energy
    // //NSEL integer number of selections
    // //NTRY number of attempts to find a second molecule
    // //CVR product of collision cross-section and relative speed
    // //PROB a probability
    // //KT third body molecule code
    // //ECTOT energy added at recmbination
    // //IREC initially 0, becomes 1 of a recombination occurs
    // //WFC weighting factor in the cell
    // //IEX is the reaction that occurs (1 if only one is possible)
    // //EA activation energy
    // //NPS the number of possible electronic states
    // //NAS the number of available electronic states
    //cout<<"START COLLISIONS"<<endl;
    
       
        if((FTIME-get (CCELL , 5,N)) > (get (CCELL , 3,N)))
        {

            DTC=2.e00*get (CCELL , 3,N);
            //calculate collisions appropriate to  time DTC
            if(get (ICCELL , 2,N)>1)
            {
                //no collisions calculated if there are less than two molecules in collision cell
                NN=get (ICCELL , 3,N);
                WFC=1.e00;
                if(IWF==1 && IVB==0)
                {
                    //dout
                    WFC=1.e00+WFM*powf(get (CELL , 1,NN),IFX);
                }
                get (CCELL , 5,N)=get (CCELL , 5,N)+DTC ;
                if(IVB==0)
                {
                    AAA=get (CCELL , 1,N);
                }
                if(IVB==1)
                {
                    C=(XB[2]+VELOB*FTIME-XB[1])/(double)(NDIV*NCIS);
                    //dout
                    XMIN=XB[1]+(double)(N-1)*C;
                    XMAX=XMIN+C;
                    //dout
                    WFC=1.e00+WFM*powf((0.5e00*(XMIN+XMAX)),IFX);
                    if(IFX==0)
                    {
                        AAA=XMAX-XMIN;
                    }
                    if(IFX==1)
                    {
                        AAA=PI*(powf(XMAX,2)-powf(XMIN,2)); //assumes unit length of full cylinder
                    }
                    if(IFX==2)
                    {
                        AAA=1.33333333333333333333e00*PI*(powf(XMAX,3)-powf(XMIN,3));    //flow is in the full sphere
                    }
                }
                //these statements implement the N(N-1) scheme
                ASEL=0.5e00*get (ICCELL , 2,N)*(get (ICCELL , 2,N)-1)*WFC*FNUM*get (CCELL , 4,N)*DTC/AAA+get (CCELL , 2,N);
                NSEL=ASEL;
                //dout
                get (CCELL , 2,N)=ASEL-(double)(NSEL);
                if(NSEL>0)
                {
                    I=0; //counts the number of selections
                    KL=0; //becomes 1 if it is the last selection
                    IIII=0; //becomes 1 if there is a recombination
                    for(KL=1;KL<=NSEL;KL++)
                    {
                        I=I+1;
                        III=0; //becomes 1 if there is no valid collision partner
                        if(get (ICCELL , 2,N)==2)
                        {
                            K=1+get (ICCELL , 1,N);
                            //dout
                            L=ICREF[K];
                            K=2+get (ICCELL , 1,N);
                            //dout
                            M=ICREF[K];
                            if(M==IPCP[L])
                            {
                                III=1;
                                get (CCELL , 5,N)=get (CCELL , 5,N)-DTC;
                            }
                        }
                        else
                        {
                            //dout
                            //                            RANDOM_NUMBER(RANF);
                            RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                            K=(int)(RANF*(double)(get (ICCELL , 2,N)))+get (ICCELL , 1,N)+1;
                            //dout
                            L=ICREF[K];
                            //one molecule has been selected at random
                            if(NNC==0)
                            {
                                //select the collision partner at random
                                M=L;
                                NTRY=0;
                                while(M==L)
                                {
                                    //dout
                                    //                                    RANDOM_NUMBER(RANF);
                                    RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                    K=(int)(RANF*(double)(get (ICCELL , 2,N)))+get (ICCELL , 1,N)+1;
                                    M=ICREF[K];
                                    if(M==IPCP[L])
                                    {
                                        if(NTRY<5*get (ICCELL , 2,N))
                                        {
                                            M=L;
                                        }
                                        else
                                        {
                                            III = 1;
                                            get (CCELL , 5,N)=get (CCELL , 5,N)-DTC/ASEL;
                                            M=L+1;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                //elect the nearest from the total number (< 30) or a random 30
                                if(get (ICCELL , 2,N)<30)
                                {
                                    LL=get (ICCELL , 2,N);
                                }
                                else
                                {
                                    LL=30;
                                }
                                SEP=1.0e10;
                                M=0;
                                for(J=1;J<=LL;J++)
                                {
                                    if(LL<30)
                                    {
                                        K=J+get (ICCELL , 1,N);
                                    }
                                    else
                                    {
                                        //                                        RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        K=(int)(RANF*(double)(get (ICCELL , 2,N)))+get (ICCELL , 1,N)+1;
                                    }
                                    MM=ICREF[K];
                                    if(MM != L)
                                    {
                                        //exclude the already selected molecule
                                        if(MM != IPCP[L])
                                        {
                                            //exclude the previous collision partner
                                            //dout
                                            A=fabsf(get (PX , 1,L)-get (PX , 1,MM));
                                            if(A<SEP&& A>1.e-8*DDIV)
                                            {
                                                M=MM;
                                                SEP=A;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if(III==0)
                        {
                            for(KK=1;KK<=3;KK++)
                            {
                                VRC[KK]=get(PV  ,KK,L)-get(PV  ,KK,M);
                            }
                            VRR=VRC[1]*VRC[1]+VRC[2]*VRC[2]+VRC[3]*VRC[3];
                            VR=sqrtf(VRR);
                            VRI=VR;
                            //Simple GAs
                            if(MSP==1)
                            {
                                //dout
                                CVR=VR*CXSS*powf(2.e00*BOLTZ*get (SP , 2,1)/(RMAS*VRR),(get (SP , 3,1)-0.5e00))*RGFS;
                                if(CVR>get (CCELL , 4,N))
                                {
                                    get (CCELL , 4,N)=CVR;
                                }
                                //dout
                                //      RANDOM_NUMBER(RANF);
                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                if(RANF<CVR/get (CCELL , 4,N))
                                {
                                    // the collision occurs
                                    if(M==IPCP[L]&& L==IPCP[M])
                                    {
                                        //file_9<<"Duplicate collision"<<endl;
                                    }
                                    //atomicAdd(&TOTCOL,1.e00);
                                    //TOTCOL=TOTCOL+1.e00;
                                    get (COLL_TOTCOL , N)=get (COLL_TOTCOL , N)+1.e00;    //problem
                                    get (TCOL , 1,1)=get (TCOL , 1,1)+2.e00;    //problem
                                    COLLS[NN]=COLLS[NN]+1.e000;     //problem
                                    WCOLLS[NN]=WCOLLS[NN]+WFC;
                                    //dout
                                    SEP=fabsf(get (PX , 1,L)-get (PX , 1,M));
                                    CLSEP[NN]=CLSEP[NN]+SEP;
                                    if(get(ISPR ,1,1)>0)
                                    {
                                        //Larsen-Borgnakke serial redistribution
                                        ECT=0.5e00*RMAS*VRR;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            //consider the molecules in turn
                                            if(NSP==1)
                                            {
                                                K=L;
                                            }
                                            else
                                            {
                                                K=M;
                                            }
                                            if(MMVM>0)
                                            {
                                                if(get(ISPV  ,1)>0)
                                                {
                                                    for(KV=1;KV<=get(ISPV  ,1);KV++)
                                                    {
                                                        EVIB=(double)(get (IPVIB , KV,K)*BOLTZ*get(SPVM ,1,KV,1));
                                                        ECC=ECT+EVIB;
                                                        if(get(SPVM ,3,KV,1)>0.0)
                                                        {
                                                            MAXLEV=ECC/(BOLTZ*get(SPVM ,1,KV,1));
                                                            B=get(SPVM ,4,KV,1)/get(SPVM ,3,KV,1); //Tdiss/Tref
                                                            A= get(SPVM ,4,KV,1)/get (VAR , 8,NN) ;//Tdiss/Ttrans
                                                            //ZV=(A*SPM(3,1,1))*(SPVM(3,KV,1)*(B*(-SPM(3,1,1))))*(((A*0.3333333D00)-1.D00)/((B*0.33333D00)-1.D00))
                                                            ZV=powf(A,get (SPM , 3,1,1))*powf(get(SPVM ,3,KV,1)*powf(B,-get (SPM , 3,1,1)),((powf(A,0.3333333e00)-1e00)/(powf(B,33333e00)-1.e00)));
                                                        }
                                                        else
                                                        {
                                                            ZV=get(SPVM ,2,KV,1);
                                                            MAXLEV=ECC/(BOLTZ*get(SPVM ,1,KV,1))+1;
                                                        }
                                                        //dout
                                                        //                                                        RANDOM_NUMBER(RANF);
                                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                        if(1.e00/ZV>RANF)
                                                        {
                                                            II=0;
                                                            while(II==0)
                                                            {
                                                                //dout
                                                                //                                                                RANDOM_NUMBER(RANF);
                                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                IV=RANF*(MAXLEV+0.99999999e00);
                                                                get (IPVIB , KV,K)=IV;
                                                                EVIB=(double)(IV)*BOLTZ;
                                                                if(EVIB<ECC)
                                                                {
                                                                    PROB=powf((1.e00-EVIB/ECC),(1.5e00-get (SPM , 3,KV,1)));
                                                                    //PROB is the probability ratio of eqn (3.28)
                                                                    //dout
                                                                    //                                                                    RANDOM_NUMBER(RANF);
                                                                    RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                    if(PROB>RANF)
                                                                        II=1;
                                                                }
                                                            }
                                                            ECT=ECC-EVIB;
                                                        }
                                                    }
                                                }
                                            }
                                            //now rotation of this molecule
                                            //dout
                                            if(get(ISPR ,1,1) > 0)
                                            {
                                                if(get(ISPR ,2,1)==0)
                                                {
                                                    B=1.e00/get(SPR ,1,1);
                                                }
                                                else //use molecule rather than mean value
                                                {
                                                    B=1.e00/(get(SPR ,1,1)+get(SPR ,2,1)*get (VAR , 8,NN)+get(SPR ,3,1)*powf(get (VAR , 8,NN),2));
                                                }
                                                //dout
                                                //                                                RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                if(B>RANF)
                                                {
                                                    ECC=ECT +PROT[K];
                                                    if(get(ISPR ,1,1)==2)
                                                    {
                                                        //dout
                                                        //                                                        RANDOM_NUMBER(RANF);
                                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                        ERM=1.e00-powf(RANF,1.e00/(2.5e00-get (SP , 3,1))); //eqn(5.46)
                                                    }
                                                    else
                                                    {
                                                        //dout
                                                        LBS(0.5e00*get(ISPR ,1,1)-1.e00,1.5e00-get (SP , 3,1),ERM);
                                                    }
                                                    PROT[K]=ERM*ECC;
                                                    ECT=ECC-PROT[K];
                                                }
                                            }
                                        }
                                        //adjust VR for the change in energy;
                                        VR=sqrtf(2.e00*ECT/get (SPM , 1,1,1));
                                    }
                                    //end of L-B redistribution
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        VCM[KK]=0.5e00*(get(PV  ,KK,L)+get(PV  ,KK,M));
                                    }
                                    //dout
                                    if(fabsf(get (SP , 4,1)-1.0) < 0.001)
                                    {
                                        //use the VHS logic //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*RANF-1.e00;
                                        //B is the cosine of a random elevation angle
                                        A=sqrtf(1.e00-B*B);
                                        VRCP[1]=B*VR;
                                        //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*RANF;
                                        //C is a random azimuth angle
                                        //dout
                                        VRCP[2]=A*cos(C)*VR;
                                        VRCP[3]=A*sin(C)*VR;
                                    }
                                    else
                                    {
                                        //use the VSS logic //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*(powf(RANF,get (SP , 4,1)))-1.e00;
                                        //B is the cosine of the deflection angle for the VSS model (Eqn. 11.8) of Bird(1994))
                                        A=sqrtf(1.e00-B*B);
                                        //dout
                                        //                                                 RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*RANF;
                                        //dout
                                        OC=(double)cos(C);
                                        SD=(double)sin(C);
                                        D=sqrtf(powf(VRC[2],2)+powf(VRC[3],2));
                                        VRA=VR/VRI;
                                        VRCP[1]=(B*VRC[1]+A*SD*D)*VRA;
                                        VRCP[2]=(B*VRC[2]+A*(VRI*VRC[3]*OC-VRC[1]*VRC[2]*SD)/D)*VRA;
                                        VRCP[3]=(B*VRC[2]+A*(VRI*VRC[2]*OC-VRC[1]*VRC[3]*SD)/D)*VRA;
                                        //the post-collision rel. velocity components are based on eqn (3.18)
                                    }
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        get(PV  ,KK,L)=VCM[KK]+0.5e00*VRCP[KK];
                                        get(PV  ,KK,M)=VCM[KK]-0.5e00*VRCP[KK];
                                    }
                                    IPCP[L]=M;
                                    IPCP[M]=L;
                                }
                            } //collision occurrence
                            
                            else
                            {
                                //Gas Mixture
                                LS=fabsf(get(IPSP ,L));
                                MS=fabsf(get(IPSP ,M));
                                CVR=VR*get (SPM , 2,LS,MS)*powf(((2.e00*BOLTZ*get (SPM , 5,LS,MS))/((get (SPM , 1,LS,MS))*VRR)),(get (SPM , 3,LS,MS)-0.5e00))*get (SPM , 6,LS,MS);
                                if(CVR>get (CCELL , 4,N))
                                {
                                    get (CCELL , 4,N)=CVR;
                                }
                                //dout
                                //                                    RANDOM_NUMBER(RANF);
                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                if(RANF<CVR/get (CCELL , 4,N) && get(IPCELL  ,L)>0 && get(IPCELL  ,M)>0)
                                {
                                    //the collision occurs (-ve IPCELL indicates recombined molecule marled for removal)
                                    if(M==IPCP[L] && L==IPCP[M])
                                    {
                                        //file_9<<"Duplicate collision";
                                    }
                                    //atomicAdd(&TOTCOL,1.e00);
                                    //TOTCOL=TOTCOL+1.e00;
                                    get (COLL_TOTCOL , N)=get (COLL_TOTCOL , N)+1.e00;
                                    get (TCOL , LS,MS)=get (TCOL , LS,MS)+1.e00;
                                    get (TCOL , MS,LS)=get (TCOL , MS,LS)+1.e00;
                                    COLLS[NN]=COLLS[NN]+1.e00;
                                    WCOLLS[NN]=WCOLLS[NN]+WFC;
                                    SEP=fabsf(get (PX , 1,L)-get (PX , 1,M));
                                    CLSEP[NN]=CLSEP[NN]+SEP;
                                    RML=get (SPM , 1,LS,MS)/get (SP , 5,MS);
                                    RMM=get (SPM , 1,LS,MS)/get (SP , 5,LS);
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        VCM[KK]=RML*get(PV  ,KK,L)+RMM*get(PV  ,KK,M);
                                    }
                                    IDISS=0;
                                    IREC=0;
                                    IEX=0;
                                    //check for dissociation
                                    if(get(ISPR ,1,LS)>0 || get(ISPR ,1,MS)>0)
                                    {
                                        ECT=0.5e00*get (SPM , 1,LS,MS)*VRR;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            if(NSP==1)
                                            {
                                                K=L; KS=LS; JS=MS;
                                            }
                                            else
                                            {
                                                K=M ; KS=MS ; JS=LS;
                                            }
                                            if(MMVM>0)
                                            {
                                                if(get(ISPV  ,KS)>0)
                                                {
                                                    for(KV=1;KV<=get(ISPV  ,KS);KV++)
                                                    {
                                                        if(get (IPVIB , KV,K)>=0 && IDISS==0)
                                                        {
                                                            //do not redistribute to a dissociating molecule marked for removal
                                                            EVIB=(double)(get (IPVIB , KV,K)*BOLTZ*get(SPVM ,1,KV,KS));
                                                            ECC=ECT+EVIB;
                                                            MAXLEV=ECC/(BOLTZ*get(SPVM ,1,KV,KS));
                                                            LIMLEV=get(SPVM ,4,KV,KS)/get(SPVM ,1,KV,KS);
                                                            if(MAXLEV > LIMLEV)
                                                            {
                                                                //dissociation occurs subject to reduction factor  -  reflects the infinity of levels past the dissociation limit
                                                                //dout
                                                                //                                                                    RANDOM_NUMBER(RANF)
                                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                if(RANF<get(SPVM ,5,KV,KS))
                                                                {
                                                                    IDISS=1;
                                                                    LZ=get (IPVIB , KV,K);
                                                                    //NDISSL[LZ]=NDISSL[LZ]+1;
                                                                    ECT=ECT-BOLTZ*get(SPVM ,4,KV,KS)+EVIB;
                                                                    //adjust VR for the change in energy
                                                                    VRR=2.e00*ECT/get (SPM , 1,LS,MS);
                                                                    VR=sqrtf(VRR);
                                                                    get (IPVIB , KV,K)=-1;
                                                                    //a negative IPVIB marks a molecule for dissociation
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    IEX=0;    //becomes the reaction number if a reaction occurs
                                    IREC=0;   //becomes 1 if a recombination occurs
                                    if(IDISS==0)
                                    {
                                        //dissociation has not occurred
                                        //consider possible recombinations
                                        if(get(ISPRC ,LS,MS)>0 && get (ICCELL , 2,N)>2)
                                        {
                                            //possible recombination using model based on collision volume for equilibrium
                                            KT=L;
                                            //NTRY=0
                                            while(KT==L||KT==M)
                                            {
                                                NTRY+=1;
                                                // if(NTRY>100)
                                                // {
                                                //  cout>>"NTRY 3rd body"<<NTRY;
                                                // }
                                                //RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);\
                                                K=(int)(RANF*(double)(get (ICCELL , 2,N]))+get (ICCELL , 1,N]+1;
                                                KT=ICREF[K];
                                            }
                                            KS=get(IPSP ,KT);
                                            //the potential third body is KT OF species KS
                                            AA=(PI/6.e00)*powf((get (SP , 1,LS)+get (SP , 1,MS)+get (SP , 1,KS)),3); //reference volume
                                            BB=AA*get( SPRC ,1,LS,MS,KS)*powf(get (VAR , 8,NN)/get(SPVM ,1,get(ISPRK ,LS,MS),get(ISPRC ,LS,MS)),get( SPRC ,2,LS,MS,KS));//collision volume
                                            B=BB*get (ICCELL , 2,N)*FNUM/AAA;
                                            if(B>1.e00)
                                            {
                                                printf("THREE BODY PROBABILITY %f\n", B);
                                                //cout<<"THREE BODY PROBABILITY"<<B;
                                                //for low density flows in which three-body collisions are very rare, it is advisable to consider recombinations in only a small
                                                //fraction of collisions and to increase the pribability by the inverse of this fraction.  This message provides a warning if this
                                                //factor has been set to an excessively large value
                                            }
                                            //dout
                                            //                                                RANDOM_NUMBER(RANF);
                                            RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                            if(RANF<B)
                                            {
                                                IREC=1;
                                                TRECOMB[get(ISPRC ,LS,MS)]=TRECOMB[get(ISPRC ,LS,MS)]+1.e00;
                                                //the collision now becomes a collision between these with L having the center of mass velocity
                                                A=0.5e00*get (SPM , 1,LS,MS)*VRR ;//the relative energy of the recombining molecules
                                                if(get(ISPR ,1,LS)>0)
                                                    A=A+PROT[L];
                                                if(MELE>1)
                                                    A=A+PELE[L];
                                                if(get(ISPV  ,LS)>0)
                                                {
                                                    for(KVV=1;KVV<=get(ISPV  ,LS);KVV++)
                                                    {
                                                        JI=get (IPVIB , KVV,L);
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        A=A+(double)(JI)*BOLTZ*get(SPVM ,1,KVV,LS);
                                                    }
                                                }
                                                if(get(ISPR ,1,MS)>0)
                                                    A+=PROT[M];
                                                if(MELE>1)
                                                    A=A+PELE[M];
                                                if(get(ISPV  ,MS)>0)
                                                {
                                                    for(KVV=1;KVV<=get(ISPV  ,MS);KVV++)
                                                    {
                                                        JI=get (IPVIB , KVV,M);
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        A=A+(double)(JI)*BOLTZ*get(SPVM ,1,KVV,MS);
                                                    }
                                                }
                                                get (TREACL , 2,LS)=get (TREACL , 2,LS)-1;
                                                get (TREACL , 2,MS)=get (TREACL , 2,MS)-1;
                                                LSI=LS;
                                                MSI=MS;
                                                LS=get(ISPRC ,LS,MS);
                                                get(IPSP ,L)=LS;
                                                //any additional vibrational modes must be set to zero
                                                IVM=get(ISPV  ,LSI);
                                                NMC=get(IPSP ,L);
                                                NVM=get(ISPV  ,NMC);
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        get (IPVIB , KV,L)=0;
                                                    }
                                                }
                                                if(MELE>1)
                                                    PELE[KV]=0.e00;

                                                get(IPCELL  ,M) = -100; //recombining molecule M marked for removal
                                                M=KT; //third body molecule is set as molecule M
                                                MS=KS;
                                                get (TREACG , 2,LS)=get (TREACG , 2,LS)+1;
                                                if(get(ISPR ,1,LS)>0)
                                                {
                                                    PROT[L]=0.e00;
                                                }
                                                if(MELE>1)
                                                    PELE[L]=0.e00;
                                                if(get(ISPV  ,LS)>0)
                                                {
                                                    for(KVV=1;KVV<=get(ISPV  ,LS);KVV++)
                                                    {
                                                        if(get (IPVIB , KVV,L)<0)
                                                        {
                                                            get (IPVIB , KVV,L)=-99999;
                                                        }
                                                        else
                                                        {
                                                            get (IPVIB , KVV,L)=0;
                                                        }
                                                    }
                                                }
                                                if(get(ISPR ,1,MS)>0)
                                                {
                                                    PROT[M]=PROT[KT];
                                                }
                                                if(MELE>1)
                                                    PELE[M]=PELE[KT];
                                                if(get(ISPV  ,MS)>0)
                                                {
                                                    for(KVV=1;KVV<=get(ISPV  ,MS);KVV++)
                                                    {
                                                        get (IPVIB , KVV,M)=get (IPVIB , KVV,KT);
                                                    }
                                                }
                                                ECTOT=A+get(SPVM ,4,1,LS)*BOLTZ ; //the energy added to this collision
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    get(PV  ,KK,L)=VCM[KK];
                                                }
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    VRC[KK]=get(PV  ,KK,L)-get(PV  ,KK,M);
                                                }
                                                VRR=VRC[1]*VRC[1]+VRC[2]*VRC[2]+VRC[3]*VRC[3];
                                                ECT=0.5e00*get (SPM , 1,LS,MS)*VRR*ECTOT;
                                                //set the vibrational energy of the recombined molecule L to enforce detailed balance
                                                IK=-1;
                                                NK=-1;
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                //NTRY=0;
                                                while(IK<0)
                                                {
                                                    // NTRY+=1;
                                                    // if(NTRY>100)
                                                    //   cout<<"NTRY VibEn"<<NTRY;
                                                    NK=NK+1;
                                                    BB=(get (VAR , 8,NN)-get( SPRT ,1,LSI,MSI))*(get (SPRP , 2,LSI,MSI,NK+1)-get (SPRP , 1,LSI,MSI,NK+1))/(get( SPRT ,2,LSI,MSI)-get( SPRT ,1,LSI,MSI))-get (SPRP , 1,LSI,MSI,NK+1);
                                                    if(RANF<BB)
                                                        IK=NK;
                                                }
                                                get (IPVIB , 1,L)=IK;
                                                ECT=ECT-(double)(IK)*BOLTZ*get(SPVM ,1,get(ISPRK ,LSI,MSI),LS);
                                                VRR=2.e00*ECT/get (SPM , 1,LS,MS);
                                                VR=sqrtf(VRR);
                                                RML=get (SPM , 1,LS,MS)/get (SP , 5,MS);
                                                RMM=get (SPM , 1,LS,MS)/get (SP , 5,LS);
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    VCM[KK]=RML*get(PV  ,KK,L)+RMM*get(PV  ,KK,M);
                                                }
                                            }
                                        }
                                        //consider exchange and chain reactions
                                        if(get(NSPEX  ,LS,MS)>0 && IREC==0 && IDISS==0)
                                        {
                                            //possible exchange reaction
                                            //memget(PSF,0.e00,sizeof(*PSF));//PSF=0.e00; //PSF(MMEX) PSF is the probability that this reaction will occur in this collision
                                            for(int i=0;i<MMEX+1;i++)
                                                PSF[i]=0.e00;
                                            
                                            for(JJ=1;JJ<=get(NSPEX  ,LS,MS);JJ++)
                                            {
                                                if(LS==get(ISPEX  ,JJ,1,LS,MS))
                                                {
                                                    K=L; KS=LS;JS=MS;
                                                }
                                                else
                                                {
                                                    K=M; KS=MS; JS=LS;
                                                }
                                                //the pre-collision molecule that splits is K of species KS
                                                if(get(SPEX  ,3,JJ,LS,MS)<0.e00)
                                                    KV=get(ISPEX  ,JJ,5,LS,MS);
                                                if(get(SPEX  ,3,JJ,LS,MS)>0.e00)
                                                {
                                                    KV=get(ISPEX  ,JJ,7,LS,MS);
                                                }
                                                JI=get (IPVIB , KV,K);
                                                if(JI<0)
                                                    JI=-JI;
                                                if(JI==99999)
                                                    JI=0;
                                                ECC=0.5e00*get (SPM , 1,LS,MS)*VRR+(double)(JI)*BOLTZ*get(SPVM ,1,KV,KS);
                                                if(get(SPEX  ,3,JJ,KS,JS)>0.e00)
                                                {
                                                    //reverse exothermic reaction
                                                    PSF[JJ]=(get(SPEX  ,1,JJ,KS,JS)*powf(get (VAR , 8,NN)/273.e00,get(SPEX  ,2,JJ,KS,JS)))*expf(-get(SPEX  ,6,JJ,KS,JS)/(BOLTZ*get (VAR , 8,NN)));
                                                }
                                                else
                                                {
                                                    //forward endothermic reaction
                                                    MAXLEV=ECC/(BOLTZ*get(SPVM ,1,KV,KS));
                                                    EA=fabsf(get(SPEX  ,3,JJ,KS,JS)); //temporarily just the heat of reaction;
                                                    if(ECC>EA)
                                                    {
                                                        //the collision energy must exceed the heat of reaction
                                                        EA=EA+get(SPEX  ,6,JJ,KS,JS); //the activation energy now includes the energy barrier
                                                        DEN=0.e00;
                                                        for(IAX=0;IAX<=MAXLEV;IAX++)
                                                        {
                                                            DEN=DEN+powf((1.e00-(double)(IAX)*BOLTZ*get(SPVM ,1,KV,KS)/ECC),(1.5e00-get (SPM , 3,KS,JS)));
                                                        }
                                                        PSF[JJ]=(double)(get(ISPEX  ,JJ,6,LS,MS))*powf((1.e00-EA/ECC),(1.5e00-get (SPM , 3,KS,JS)))/DEN;
                                                    }
                                                }
                                            }
                                            if(get(NSPEX  ,LS,MS)>1)
                                            {
                                                BB=0.e00;
                                                for(JJ=1;JJ<=get(NSPEX  ,LS,MS);JJ++)
                                                {
                                                    BB=BB+PSF[JJ];
                                                }
                                                //BB is the sum of the probabilities
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                if(BB>RANF)
                                                {
                                                    BB=0.e00;
                                                    IEX=0;
                                                    JJ=0;
                                                    //NTRY=0;
                                                    while(JJ<get(NSPEX  ,LS,MS)&& IEX==0)
                                                    {
                                                        // NTRY=NTRY+1;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout<<"NTRY find IEX"<<NTRY;
                                                        // }
                                                        JJ+=1;
                                                        BB+=PSF[JJ];
                                                        if(BB>RANF)
                                                            IEX=JJ;
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                IEX=0;
                                                if(PSF[1]>RANF)
                                                    IEX=1;
                                            }
                                            if(IEX>0)
                                            {
                                                //exchange or chain reaction occurs
                                                JX=get(NEX  ,IEX,LS,MS);
                                                //cout<<"Reaction"<<JX;
                                                TNEX[JX]=TNEX[JX]+1.e00;
                                                //cout<<IEX<<L<<M<<LS<<MS;
                                                get(IPSP ,L)=get(ISPEX  ,IEX,3,LS,MS); //L is now the new molecule that splits
                                                get(IPSP ,M)=get(ISPEX  ,IEX,4,LS,MS);
                                                LSI=LS;
                                                MSI=MS;
                                                //any additional vibrational modes must be set to zero
                                                IVM=get(ISPV  ,LS);
                                                NMC=IPCP[L];
                                                NVM=get(ISPV  ,NMC);
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        get (IPVIB , KV,L)=0;
                                                    }
                                                }
                                                IVM=get(ISPV  ,MS);
                                                NMC=IPCP[M];
                                                NVM=get(ISPV  ,NMC);
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        get (IPVIB , KV,M)=0;
                                                    }
                                                }
                                                //put all pre-collision energies into the relative translational energy and adjust for the reaction energy
                                                ECT=0.5e00*get (SPM , 1,LS,MS)*VRR;
                                                if(get(ISPR ,1,LS)>0)
                                                    ECT=ECT+PROT[L];
                                                if(MELE>1)
                                                    ECT=ECT+PELE[L];
                                                if(get(ISPV  ,LS)>0)
                                                {
                                                    for(KV=1;KV<=get(ISPV  ,LS);KV++)
                                                    {
                                                        JI=get (IPVIB , KV,L);
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        ECT=ECT+(double)(JI)*BOLTZ*get(SPVM ,1,KV,LS);
                                                    }
                                                }
                                                if(get(ISPR ,1,MS)>0)
                                                    ECT=ECT+PROT[M];
                                                if(get(ISPR ,1,MS)) AA = PROT[M] ;
                                                if(MELE>1)
                                                    ECT=ECT+PELE[M];
                                                if(get(ISPV  ,MS)>0)
                                                {
                                                    for(KV=1;KV<=get(ISPV  ,MS);KV++)
                                                    {
                                                        JI=get (IPVIB , KV,M);
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        ECT=ECT+(double)(JI)*BOLTZ*get(SPVM ,1,KV,MS);
                                                    }
                                                }
                                                ECT=ECT+get(SPEX  ,3,IEX,LS,MS);
                                                if(ECT<0.0)
                                                {
                                                    //printf ("-VE ECT %f\n",ECT);
                                                    //printf ("REACTION %d",JJ," BETWEEN %d",LS," & %d\n",MS);
                                                    // cout<<"-VE ECT "<<ECT<<endl;
                                                    // cout<<"REACTION "<<JJ<<" BETWEEN "<<LS<<" "<<MS<<endl;
                                                    //dout
                                                    //cin.get();
                                                    return ;
                                                }
                                                if(get(SPEX  ,3,IEX,LS,MS)<0.e00)
                                                {
                                                    get (TREACL , 3,LS)=get (TREACL , 3,LS)-1;
                                                    get (TREACL , 3,MS)=get (TREACL , 3,MS)-1;
                                                    LS=get(IPSP ,L) ;
                                                    MS=get(IPSP ,M) ;
                                                    get (TREACG , 3,LS)=get (TREACG , 3,LS)+1;
                                                    get (TREACG , 3,MS)=get (TREACG , 3,MS)+1;
                                                }
                                                else
                                                {
                                                    get (TREACL , 4,LS)=get (TREACL , 4,LS)-1;
                                                    get (TREACL , 4,MS)=get (TREACL , 4,MS)-1;
                                                    LS=get(IPSP ,L) ;
                                                    MS=get(IPSP ,M) ;
                                                    get (TREACG , 4,LS)=get (TREACG , 4,LS)+1;
                                                    get (TREACG , 4,MS)=get (TREACG , 4,MS)+1;
                                                }
                                                RML=get (SPM , 1,LS,MS)/get (SP , 5,MS);
                                                RMM=get (SPM , 1,LS,MS)/get (SP , 5,LS);
                                                //calculate the new VRR to match ECT using the new molecular masses
                                                VRR=2.e00*ECT/get (SPM , 1,LS,MS);
                                                if(get(ISPV  ,LS)>0)
                                                {
                                                    for(KV=1;get(ISPV  ,LS);KV++)
                                                    {
                                                        if(get (IPVIB , KV,L)<0)
                                                        {
                                                            get (IPVIB , KV,L)=-99999;
                                                        }
                                                        else
                                                        {
                                                            get (IPVIB , KV,L)=0;
                                                        }
                                                    }
                                                }
                                                if(get(ISPR ,1,LS)>0)
                                                    PROT[L]=0;
                                                if(MELE>1)
                                                    PELE[L]=0.e00;
                                                if(get(ISPV  ,MS)>0)
                                                {
                                                    for(KV=1;get(ISPV  ,MS);KV++)
                                                    {
                                                        if(get (IPVIB , KV,M)<0)
                                                        {
                                                            get (IPVIB , KV,M)=-99999;
                                                        }
                                                        else
                                                        {
                                                            get (IPVIB , KV,M)=0;
                                                        }
                                                    }
                                                }
                                                if(get(ISPR ,1,MS)>0)
                                                    PROT[M]=0;
                                                if(MELE>1)
                                                    PELE[M]=0.e00;
                                                //set vibrational level of product molecule in exothermic reaction to enforce detailed balance
                                                if(get(SPEX  ,3,IEX,LSI,MSI)>0.e00)
                                                {
                                                    //exothermic exchange or chain reaction
                                                    IK=-1; //becomes 0 when the level is chosen
                                                    NK=-1;
                                                    //dout
                                                    //                                                        RANDOM_NUMBER(RANF);
                                                    RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                    //NTRY=0;
                                                    while(IK<0)
                                                    {
                                                        // NTRY=NTRY+1;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout>>"NTRY VibProd"<<NTRY<<endl;
                                                        // }
                                                        NK=NK+1;
                                                        BB=(get (VAR , 8,NN)-get(SPEX  ,4,IEX,LSI,MSI))*  (get (SPREX , 2,IEX,LSI,MSI,NK+1)-get (SPREX , 1,IEX,LSI,MSI,NK+1))/(get(SPEX  ,5,IEX,LSI,MSI)-get(SPEX  ,4,IEX,LSI,MSI))+get (SPREX , 1,IEX,LSI,MSI,NK+1);
                                                        if(RANF<BB)
                                                            IK=NK;
                                                    }
                                                    if(get(NSLEV ,1 , LS)>0)
                                                    {
                                                        IK+=get(NSLEV ,1,LS);
                                                        get(NSLEV ,1,LS)=0;
                                                    }
                                                    KV=get(ISPEX  ,IEX,7,LSI,MSI);
                                                    get (IPVIB , KV,L)=IK;
                                                    EVIB=(double)(IK)*BOLTZ*get(SPVM ,1,KV,LS);
                                                    ECT=ECT-EVIB;
                                                    if(ECT<0.e00)
                                                    {
                                                        //NTRY=0;
                                                        while(ECT<0.e00)
                                                        {
                                                            //NTRY+=1;
                                                            // if(NTRY>100)
                                                            //     cout<<"NTRY ECT<0"<<NTRY<<endl;
                                                            get (IPVIB , KV,L)=get (IPVIB , KV,L)-1;
                                                            get(NSLEV ,1,LS)+=1;
                                                            ECT=ECT+BOLTZ*get(SPVM ,1,KV,LS);
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    //for endothermic reaction, select vibration from vib. dist. at macroscopic temperature
                                                    //normal L-B selection would be from the excessively low energy after the endo. reaction
                                                    KV=get(ISPEX  ,IEX,5,LS,MS);
                                                    //dout
                                                    SVIB( LS,get (VAR , 8,NN),IK,KV);
                                                    if(get(NSLEV ,2,LS)>0)
                                                    {
                                                        IK=IK+get(NSLEV ,2,LS);
                                                        get(NSLEV ,2,LS)=0;
                                                    }
                                                    get (IPVIB , KV,L)=IK;
                                                    EVIB=(double)(IK)*BOLTZ*get(SPVM ,1,KV,LS);
                                                    ECT=ECT-EVIB;
                                                    if(ECT<0.e00)
                                                    {
                                                        //NTRY=0;
                                                        while(ECT<0.e00)
                                                        {
                                                            //NTRY+=1;
                                                            get (IPVIB , KV,L)-=1;
                                                            get(NSLEV ,2,LS)+=1;
                                                            ECT=ECT+BOLTZ*get(SPVM ,1,KV,LS);
                                                            // if(NTRY>100)
                                                            // {
                                                            //cout<<"NTRY ECT<0#2"<<NTRY<<endl;
                                                            // get (IPVIB , KV,L]=0;
                                                            //   ECT+=EVIB;
                                                            //   NSLEV[2,LS]=0;
                                                            // }
                                                        }
                                                    }
                                                }
                                                //set rotational energy of molecule L to equilibrium at the macroscopic temperature
                                                SROT( LS,get (VAR , 8,NN),PROT[L]);
                                                if(SLER[LS]>1.e-21)
                                                {
                                                    PROT[L]+=SLER[LS];
                                                    SLER[LS]=1.e-21;
                                                }
                                                ECT-=PROT[L] ;
                                                ABA=PROT[L] ;
                                                if(ECT<0.e00)
                                                {
                                                    //NTRY=0;
                                                    while(ECT<0.e00)
                                                    {
                                                        //NTRY+=1;
                                                        BB=0.5e00*PROT[L];
                                                        SLER[LS]+=BB;
                                                        PROT[L]=BB;
                                                        ECT+=BB;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout<<"NTRY ECT<0#3"<<NTRY<<L<<endl;
                                                        //   ECT+=ABA;
                                                        //   PROT[L]=0;
                                                        //   SLER[LS]=1.e-21;
                                                        // }
                                                    }
                                                }
                                                //calculate the new VRR to match ECT using the new molecular masses
                                                VRR=2.e00*ECT/get (SPM , 1,LS,MS);
                                            }
                                        }
                                    }
                            
                                        //end of reactions other than the deferred dissociation action in the DISSOCIATION subroutine
                                    if(IREC==0 && IDISS==0)
                                    {
                                        //recombined redistribution already made and there is a separate subroutine for dissociation
                                        //Larsen-Borgnakke serial redistribution
                                        ECT=0.5e00*get (SPM , 1,LS,MS)*VRR ;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            if(NSP==1)
                                            {
                                                K=L;KS=LS;JS=MS;
                                            }
                                            else
                                            {
                                                K=M; KS=MS; JS=LS;
                                            }
                                            //now electronic energy for this molecule
                                            if(MELE>1)
                                            {
                                                B=1.e00/get(QELC ,3,1,KS);
                                                //dout
                                                //RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                if(B>RANF)
                                                {
                                                    NPS=0;
                                                    ECC=ECT+PELE[K];
                                                    if(get(NELL  ,KS)==1){
                                                        NPS=get(QELC ,1,1,KS); //number of possible states is at least the degeneracy of the ground state
                                                    }
                                                    if(get(NELL  ,KS)>1)
                                                    {
                                                        for(NEL=1;NEL<=get(NELL  ,KS);NEL++)
                                                        {
                                                            if(ECC>BOLTZ*get(QELC ,2,NEL,KS))
                                                                NPS=NPS+get(QELC ,1,NEL,KS);
                                                        }
                                                        II=0;
                                                        //NTRY=0;
                                                        while(II==0)
                                                        {
                                                            //NTRY+=1;
                                                            // if(NTRY>100)
                                                            //           cout<<"NTRY ElecEn"<<NTRY<<endl;
                                                            //dout
                                                            //                                                                    RANDOM_NUMBER(RANF);
                                                            RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                            NSTATE=ceil(RANF*NPS);//random state, now determine the energy level
                                                            NAS=0;
                                                            NLEVEL=-1;
                                                            for(NEL=1;NEL<=get(NELL  ,KS);NEL++)
                                                            {
                                                                NAS= NAS+get(QELC ,1,NEL,KS);
                                                                if(NSTATE<=NAS && NLEVEL<0)
                                                                    NLEVEL=NEL;
                                                            }
                                                            //dout
                                                            //                                                                    RANDOM_NUMBER(RANF);
                                                            RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                            if((1.e00/(B*get(QELC ,3,NLEVEL,KS)))<RANF)
                                                            {
                                                                II=1;
                                                            }
                                                            else
                                                            {
                                                                if(ECC>BOLTZ*get(QELC ,2,NLEVEL,KS))
                                                                {
                                                                    PROB=powf(1.e00-BOLTZ*get(QELC ,2,NLEVEL,KS)/ECC,(1.5e00-get (SPM , 3,KS,JS)));
                                                                    //dout
                                                                    //                                                                            RANDOM_NUMBER(RANF);
                                                                    RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                    if(PROB>RANF)
                                                                    {
                                                                        II=1;
                                                                        PELE[K]=BOLTZ*get(QELC ,2,NLEVEL,KS);
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        ECT=ECC-PELE[K];
                                                    }
                                                }
                                            }
                                            //now the vibrational energy for this molecule
                                            if(MMVM>0 && IEX==0)
                                            {
                                                if(get(ISPV  ,KS)>0)
                                                {
                                                    for(KV=1;KV<=get(ISPV  ,KS);KV++)
                                                    {
                                                        if(get (IPVIB , KV,K)>=0 && IDISS==0) //do not redistribute to a dissociating molecule marked for removal
                                                        {
                                                            EVIB=(double)(get (IPVIB , KV,K))*BOLTZ*get(SPVM ,1,KV,KS);
                                                            ECC=ECT+EVIB;
                                                            MAXLEV=ECC/(BOLTZ*get(SPVM ,1,KV,KS));
                                                            if(get(SPVM ,3,KV,KS)>0.0)
                                                            {   
                                                                B=get(SPVM ,4,KV,KS)/get(SPVM ,3,KV,KS);
                                                                A=get(SPVM ,4,KV,KS)/get (VAR , 8,NN);
                                                               ZV = powf(A,get (SPM , 3,KS,JS))*powf((get(SPVM ,2,KV,KS)*powf(B,-get (SPM , 3,KS,JS))),((powf(A,0.3333333e00)-1.e00)/(powf(B,0.33333e00)-1.e00)));
                                                               
                                                            }
                                                            else
                                                                ZV=get(SPVM ,2,KV,KS);
                                                            //                                                                    RANDOM_NUMBER(RANF) //dout
                                                            RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                            if(1.e00/ZV>RANF ||IREC==1)
                                                            {
                                                                II=0;
                                                                NSTEP=0;
                                                                while(II==0 && NSTEP<100000)
                                                                {
                                                                    NSTEP+=1;
                                                                    if(NSTEP>99000)
                                                                    {
                                                                        printf("%d %f %d\n",NSTEP,ECC,MAXLEV);
                                                                        //cout<<NSTEP<<" "<<ECC<<" "<<MAXLEV<<endl;
                                                                        //dout
                                                                        return ;
                                                                    }
                                                                    //                                                                            RANDOM_NUMBER(RANF);
                                                                    RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                    IV=RANF*(MAXLEV+0.99999999e00);
                                                                    get (IPVIB , KV,K)=IV;
                                                                    EVIB=(double)(IV)*BOLTZ*get(SPVM ,1,KV,KS);
                                                                    if(EVIB<ECC)
                                                                    {
                                                                        PROB=powf(1.e00-EVIB/ECC,1.5e00-get(SPVM ,3,KS,JS));
                                                                        //PROB is the probability ratio of eqn (3.28)
                                                                        //                                                                                RANDOM_NUMBER(RANF);
                                                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                                        if(PROB>RANF)
                                                                            II=1;
                                                                    }
                                                                }
                                                                ECT=ECC-EVIB;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            //now rotation of this molecule
                                            //dout
                                            if(get(ISPR ,1,KS) > 0)
                                            {
                                                if(get(ISPR ,2,KS)==0 && get(ISPR ,2,JS)==0)
                                                {
                                                    B=1.e00/get (SPM , 7,KS,JS);
                                                }
                                                else
                                                    B=1.e00/(get(SPR ,1,KS))+get(SPR ,2,KS)*get (VAR , 8,NN)+get(SPR ,3,KS)*powf(get (VAR , 8,NN),2);
                                                //                                                        RANDOM_NUMBER(RANF);
                                                RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                if(B>RANF|| IREC==1)
                                                {
                                                    ECC=ECT+PROT[K];
                                                    if(get(ISPR ,1,KS)==2)
                                                    {
                                                        //                                                                RANDOM_NUMBER(RANF);
                                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                                        ERM=1.e00-powf(RANF,(1.e00/(2.5e00-get (SPM , 3,KS,JS))));//eqn(5.46)
                                                    }
                                                    else
                                                        LBS( 0.5e00*get(ISPR ,1,KS)-1.e00,1.5e00-get (SPM , 3,KS,JS),ERM);
                                                    PROT[K]=ERM*ECC;
                                                    ECT=ECC-PROT[K];
                                                }
                                            }
                                        }
                                        //adjust VR for the change in energy
                                        VR=sqrtf(2.e00*ECT/get (SPM , 1,LS,MS));
                                    }//end of L-B redistribution
                                    if(fabsf(get (SPM , 8,LS,MS)-1.0)<0.001)
                                    {
                                        //use the VHS logic
                                        //                                                RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*RANF-1.e00;
                                        //B is the cosine of a random elevation angle
                                        A=sqrtf(1.e00-B*B);
                                        VRCP[1]=B*VR;
                                        //                                                RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*RANF;
                                        //C is a random azimuth angle;
                                        VRCP[2]=A*(double)cos(C)*VR;
                                        VRCP[3]=A*(double)sin(C)*VR;
                                    }
                                    else
                                    {
                                        //use the VSS logic
                                        //the VRCP terms do not allow properly for the change in VR - see new book  !STILL TO BE FIXED
                                        VRA=VR/VRI;
                                        //                                                RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*powf(RANF,get (SP , 4,1))-1.e00;
                                        // B is the cosine of the deflection angle for the VSS model
                                        A=sqrtf(1.e00-B*B);
                                        //                                                RANDOM_NUMBER(RANF);
                                        RANF=((double)rand()/(double)RAND_MAX);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*RANF;
                                        OC=(double)cos(C);
                                        SD=(double)sin(C);
                                        D=sqrtf(powf(VRC[2],2)+powf(VRC[3],2));
                                        VRCP[1]=(B*VRC[1]+A*SD*D)*VRA;
                                        VRCP[2]=(B*VRC[2]+A*(VRI*VRC[3]*OC-VRC[1]*VRC[2]*SD)/D)*VRA;
                                        VRCP[3]=(B*VRC[3]+A*(VRI*VRC[2]*OC+VRC[1]*VRC[3]*SD)/D)*VRA;
                                        //the post-collision rel. velocity components are based on eqn (3.18)
                                    }
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        get(PV  ,KK,L)=VCM[KK]+RMM*VRCP[KK];
                                        get(PV  ,KK,M)=VCM[KK]-RMM*VRCP[KK];
                                    }
                                    IPCP[L]=M;
                                    IPCP[M]=L;
                                    //call energy(0,E2)
                                    // !              IF (Dfabs(E2-E1) > 1.D-14) read(*,*)
                                }////collision occurrence
                            }
                            
                            
                        }//separate simplegas / mixture coding
                    }
                }
            }
        }
    //remove any recombined atoms
    
}


void COLLISIONS()
{   
    
    int N=NCCELLS;
    
    d_allocate(N , COLL_TOTCOL) ;
    for(int i=0 ; i<N+2 ; i++)  get (COLL_TOTCOL , i)=0e00 ;
    
    
  
    for(N=1;N<=NCCELLS;N++){
        cuda_collisions(N);
    }
    
    

   

    //std::cout<<"printf: "<< duration <<'\n';
    
    for(N=1;N<=NCCELLS;N++){
        TOTCOL=TOTCOL+get (COLL_TOTCOL , N);
    }
    for(int N=1;N<=NM;N++)
    {
        if(get(IPCELL  ,N)<0)
            REMOVE_MOL(N); 
    }
    return;
} 

void SETXT()
{
    //generate TECPLOT files for displaying an x-t diagram of an unsteady flow
    //this employs ordered data, therefore the cells MUST NOT BE ADAPTED
    //N.B. some custom coding for particular problems
    //
    //
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    //OUTPUT output;
    //
    
    // IMPLICIT NONE
    //
    int N,M,IOUT;
    double A,C;
    double *VALINT;
    // REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: VALINT
    //
    //VALINT(N,M) the interpolated values at sampling cell M boundaries and extrapolated values at boundaries
    //    N=1 distance
    //    N=2 time
    //    N=3 number density
    //    N=4 radial velocity
    //    N=5 pressure (nkT)
    //    N=6 temperature
    //    N=7 h2o fraction (Sec. 7.9 only)
    //
    //the variables in VALINT may be altered for particular problems
    //
    d_allocate(7 , NCELLS+2 , VALINT) ;
    
    
    // ALLOCATE (VALINT(6,NCELLS+1),STAT=ERROR)
    //
    //777 FORMAT(12G14.6)
    //24[]
    
    //Internal options
    IOUT=0;    //0 for dimensioned output, 1 for non-dimensional output
    //
    A=1.e00;   //dt/dt for selection of v velocity component in TECPLOT to draw particle paths as "streamlines"
    //
    if(FTIME < 0.5e00*DTM){
        //Headings and zero time record
        //        IF (ERROR /= 0) THEN
        //        WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR VALINT ARRAY',ERROR
        //        ENDIF
        NLINE=1;
        file_9<< "J in tecplot file = "<<NLINE*(NCELLS+1)<<endl;
        //  WRITE (18,*) 'VARIABLES = "Distance","Time","n","u","p","T","H2O","A"'   //for combustion wave output(Sec. 7.9)
        
        file_18<<"VARIABLES = 'Distance','Time','n','u','p','T','A' "<<endl;
        file_18<<"ZONE I= "<<NCELLS+1<<", J=  (set to number of output intervals+1), F=POINT"<<endl;
        //
        for(N=1;N<=NCELLS+1;N++){
            get (VALINT , 1,N)=XB[1]+(N-1)*DDIV;    //distance
            get (VALINT , 1,N)=get (VALINT , 1,N);         //time
            get (VALINT , 2,N)=0.0;
            get (VALINT , 3,N)=FND[1];
            get (VALINT , 4,N)=0;
            get (VALINT , 5,N)=FND[1]*BOLTZ*FTMP[1];
            get (VALINT , 6,N)=FTMP[1];
            //   VALINT(7,N)=FSP(6,1)   //FSP(6 for combustion wave
            if((get (VALINT , 1,N) > XS) && (ISECS == 1)){
                get (VALINT , 3,N)=FND[2];
                get (VALINT , 5,N)=FND[2]*BOLTZ*FTMP[2];
                get (VALINT , 6,N)=FTMP[2];
                //      VALINT(7,N)=FSP(6,2)
            }
            if(IOUT == 1){
                get (VALINT , 3,N)=1.e00;
                get (VALINT , 5,N)=1.e00;
                get (VALINT , 6,N)=1.e00;
            }
            for(M=1;M<=6;M++)
                file_18<<get (VALINT , M,N)<<"\t";//WRITE (18,777) (VALINT(M,N),M=1,6),A
            file_18<<A<<endl;
        }
    }
    else{
        NLINE=NLINE+1;
        cout<<"J in tecplot file = "<<NLINE<<endl;
        if(IVB == 0) C=DDIV;
        if(IVB == 1) C=(XB[2]+VELOB*FTIME-XB[1])/double(NDIV);
        for(N=1;N<=NCELLS+1;N++){
            get (VALINT , 1,N)=XB[1]+(N-1)*C;
            get (VALINT , 2,N)=FTIME;
            if((N > 1) && (N < NCELLS+1)){
                get (VALINT , 3,N)=0.5e00*(get (VAR , 3,N)+get (VAR , 3,N-1));
                get (VALINT , 4,N)=0.5e00*(get (VAR , 5,N)+get (VAR , 5,N-1));
                get (VALINT , 5,N)=0.5e00*(get (VAR , 18,N)+get (VAR , 18,N-1));
                get (VALINT , 6,N)=0.5e00*(get (VAR , 11,N)+get (VAR , 11,N-1));
                //     VALINT(7,N)=0.5D00*(VARSP(1,N,6)+VARSP(1,N-1,6))   //H2O fraction for Sec 7.9
            }
        }
        for(N=3;N<=6;N++)
            get (VALINT , N,1)=0.5e00*(3.e00*get (VALINT , N,2)-get (VALINT , N,3));
        
        //
        for(N=3;N<=6;N++)
            get (VALINT , N,NCELLS+1)=0.5e00*(3.e00*get (VALINT , N,NCELLS)-get (VALINT , N,NCELLS-1));
        
        //
        for(N=1;N<=NCELLS+1;N++){
            if(IOUT == 1){
                get (VALINT , 1,N)=(get (VALINT , 1,N)-XB[1])/(XB[2]-XB[1]);
                get (VALINT , 2,N)=get (VALINT , 2,N)/TNORM;
                get (VALINT , 3,N)=get (VALINT , 3,N)/FND[1];
                get (VALINT , 4,N)=get (VALINT , 4,N)/VMPM;
                get (VALINT , 5,N)=get (VALINT , 5,N)/(FND[1]*BOLTZ*FTMP[1]);
                get (VALINT , 6,N)=get (VALINT , 6,N)/FTMP[1];
            }
            for(M=1;M<=6;M++)
                file_18<<get (VALINT , M,N)<<"\t";//WRITE (18,777) (get (VALINT , M,N),M=1,6),A       //
            file_18<<A<<endl;
        }
    }
    //
    return;
}