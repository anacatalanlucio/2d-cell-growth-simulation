
#include <stdio.h>
#include <time.h>
#include <string>
#include <stdlib.h>
#include <CL/cl.h>
#include <GL/freeglut.h>

using namespace std;

//OpenCL version error ignore
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

//OpenGL variables and constants 
enum cell { CANCER, HEALTHY, MEDICINE };

//1024 x 768 cell grid 
const int ROWS = 1024;
const int COLUMNS = 768;
int opengl_totalSize = (ROWS * COLUMNS);

cell a_cell[ROWS][COLUMNS];

#define FPS 5

const int opengl_cancerInit = ROWS * COLUMNS * 0.26;


int returnCall;
cl_mem readCell; //input array
cl_mem writeCell; //output array 
cl_mem colorCell;
size_t global;
size_t local;

//GPU variables 
cl_device_id gpu_device_id;
cl_context gpu_context;
cl_command_queue gpu_command_queue;
cl_program gpu_program;
cl_kernel gpu_kernel;
cl_platform_id gpu_platform_id;
cl_uint gpu_num_of_devices = 0;
cl_uint gpu_num_of_platforms = 0;

//CPU variables 
cl_device_id cpu_device_id;
cl_context cpu_context;
cl_command_queue cpu_command_queue;
cl_program cpu_program;
cl_kernel cpu_kernel;
cl_platform_id cpu_platform_id;
cl_uint cpu_num_of_devices = 0;
cl_uint cpu_num_of_platforms = 0;


//A 18-point proportional spaced font to display the number of each cell 
//since there are too many this time with 1024 x 768
void* opengl_font = GLUT_BITMAP_HELVETICA_18;


/// <summary>
/// moving my kernels from the helloOpenCL.cl kernel file to 
/// use my two functions as sources here.
/// It is easier to access the variable names. 
/// </summary>
/// 

//GPU Kernel source
const char* SourceGPUKernel = "\n\
__kernel void UpdateCellsWithGPU(__global int* readCell, __global int* writeCell)\n\
{\n\
    int gid = get_global_id(0); \n\
    \n\
    int COLUMNS = 1024;\n\
    int ROWS = 768; \n\
    \n\
    int x = gid / ROWS; \n\
    int y = gid % ROWS; \n\
    \n\
    int neighborsNum = 0; \n\
    int cellStatusBefore = 0; \n\
    int cellStatusAfter = 0; \n\
    \n\
    int CANCER = 0; \n\
    int HEALTHY = 1; \n\
    int MEDICINE = 2; \n\
    \n\
    //if the cell is either healthy or cancer  \n\
    if (readCell[x*ROWS + y] == HEALTHY || writeCell[x*ROWS + y] == CANCER)  \n\
    { \n\
    // healthy cell is with neigbours >= 6 cancer cells becomes cancer \n\
    if (readCell[x*ROWS + y] == HEALTHY)  \n\
    { \n\
        cellStatusBefore = CANCER; \n\
        cellStatusAfter = CANCER; \n\
    } \n\
    // If a cancer cell is surrounded by >= 6 medicine cells become healthy \n\
    else if (readCell[x*ROWS + y] == CANCER) \n\
    { \n\
        cellStatusBefore = MEDICINE; \n\
        cellStatusAfter = HEALTHY; \n\
    } \n\
    // Checking status of neigbours \n\
    if (x > 0 && y > 0)  \n\
    { \n\
        if (readCell[(x - 1)*ROWS + (y - 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (y > 0)  \n\
    { \n\
        if (readCell[x*ROWS + (y - 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (x < (COLUMNS - 1) && y > 0)  \n\
    { \n\
        if (readCell[(x + 1)*ROWS + (y - 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (x > 0) { \n\
        if (readCell[(x - 1)*ROWS + y] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (x < (COLUMNS - 1))  \n\
    { \n\
        if (readCell[(x + 1)*ROWS + y] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (x > 0 && y < (ROWS - 1)) \n\
    { \n\
        if (readCell[(x - 1)*ROWS + (y + 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (y < (ROWS - 1)) \n\
    { \n\
        if (readCell[x*ROWS + (y + 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    } \n\
    if (x < (COLUMNS - 1) && y < (ROWS - 1))  \n\
    { \n\
        if (readCell[(x + 1)*ROWS + (y + 1)] == cellStatusBefore) \n\
            neighborsNum++; \n\
    }\n\
    //one cell surrounded by >= 6 of either cancer, healthy or medicine will change to that state \n\
    if (neighborsNum >= 6)  \n\
    { \n\
        writeCell[x*ROWS + y] = cellStatusAfter; \n\
    } \n\
} \n\
} \n\
\n";


// CPU kernel source
const char* SourceCPUKernel = "\n\
__kernel void UpdateCellsWithCPU(__global int* colorCell)\n\
{\n\
    int gid = get_global_id(0); \n\
    \n\
    int COLUMNS = 1024; \n\
    int ROWS = 768; \n\
    \n\
    int x = gid / ROWS; \n\
    int y = gid % ROWS; \n\
    \n\
    int CANCER = 0; \n\
    int HEALTHY = 1; \n\
    int MEDICINE = 2; \n\
    \n\
}\n";


//Mouse --> injection
void onClick(int click, int state, int x, int y)
{
    if (click == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        //injecting cancer cell  with medicine makes it become healthy
        if (a_cell[x][y] == CANCER)
        {
            a_cell[x][y] = HEALTHY;
        }

        //medicine cells movng outward radially
        else
        {
            a_cell[x][y] = MEDICINE;

            if (x > 0 && y > 0)
            {
                a_cell[x - 1][y - 1] = MEDICINE;
            }
            if (y > 0)
            {
                a_cell[x][y - 1] = MEDICINE;
            }
            if (x < (ROWS - 1) && y > 0)
            {
                a_cell[x + 1][y - 1] = MEDICINE;
            }
            if (x > 0)
            {
                a_cell[x - 1][y] = MEDICINE;
            }
            if (x < (ROWS - 1))
            {
                a_cell[x + 1][y] = MEDICINE;
            }
            if (x > 0 && y < (COLUMNS - 1))
            {
                a_cell[x - 1][y + 1] = MEDICINE;
            }
            if (y < (COLUMNS - 1))
            {
                a_cell[x][y + 1] = MEDICINE;
            }
            if (x < (ROWS - 1) && y < (COLUMNS - 1))
            {
                a_cell[x + 1][y + 1] = MEDICINE;
            }

            //this is one cell 
           //(x,y+1)------(x+1, y+1)
           // '                  '
           // '                  '
           // '                  '
           //(x ,y)--------(x+1, y)
        }
    }
}


//Updating cells --> qeueue buffers, setting kernel arguments, get results back form device
int UpdateCellsWithOpenCL()
{
      returnCall = clEnqueueWriteBuffer(gpu_command_queue, readCell, CL_TRUE, 0, sizeof(int) * opengl_totalSize, a_cell, 0, NULL, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Error writing write to source array.\n");

        exit(1);
    }

    returnCall = clEnqueueWriteBuffer(gpu_command_queue, writeCell, CL_TRUE, 0, sizeof(int) * opengl_totalSize, a_cell, 0, NULL, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Error writing read to source array.\n");

        exit(1);
    }

    returnCall = 0;
    returnCall = clSetKernelArg(gpu_kernel, 0, sizeof(cl_mem), &readCell);
    returnCall |= clSetKernelArg(gpu_kernel, 1, sizeof(cl_mem), &writeCell);

    if (returnCall != CL_SUCCESS)
    {
        printf("Error setting kernel arguments. %d\n", returnCall);

        exit(1);
    }

    returnCall = clGetKernelWorkGroupInfo(gpu_kernel, gpu_device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Error getting work group info. %d\n", returnCall);

        exit(1);
    }

    global = opengl_totalSize;

    returnCall = clEnqueueNDRangeKernel(gpu_command_queue, gpu_kernel, 1, NULL, &global, &local, 0, NULL, NULL);

    if (returnCall) 
    {
        printf("Error executing GPU kernel.\n");

        return EXIT_FAILURE;
    }
    clFinish(gpu_command_queue);

    //results from device 
    returnCall = clEnqueueReadBuffer(gpu_command_queue, writeCell, CL_TRUE, 0, sizeof(int) * opengl_totalSize, a_cell, 0, NULL, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Error reading outptt array. %d\n", returnCall);

        exit(1);
    }

    return returnCall;
}



//OpenGL recursive updating call
void UpdateCells(int value)
{
    UpdateCellsWithOpenCL();
    glutPostRedisplay();
    glutTimerFunc(1000 / FPS, UpdateCells, 0);
}


//Text displaying the number of each cell 
void renderBitmapString(float x, float y, void* font, const char* string)
{
  
    const char* c;
    glRasterPos2f(x, y);

    for (c = string; *c != '\0'; c++) 
    {
        glutBitmapCharacter(font, *c);
    }
}


//OpenGL Display from Assignment 1
void Display()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, ROWS, COLUMNS, 0);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);

    int countHealthy = 0;
    int countCancer = 0;
    int countMedicine = 0;

    for (int x = 0; x < ROWS; x++)
    {
        for (int y = 0; y < COLUMNS; y++)
        {
            if (a_cell[x][y] == HEALTHY)
            {
                glColor3f(0, 0.5, 0); //green
                countHealthy++;
            }
            else if (a_cell[x][y] == CANCER)
            {
                glColor3f(1, 0, 0); //red
                countCancer++;
            }
            else if (a_cell[x][y] == MEDICINE)
            {
                glColor3f(1, 1, 0); //yellow
                countMedicine++;
            }

            glVertex2f(x, y);
            glVertex2f(x + 1, y);
            glVertex2f(x + 1, y + 1);
            glVertex2f(x, y + 1);

            //this is one cell 
           //(x,y+1)------(x+1, y+1)
           // '                  '
           // '                  '
           // '                  '
           //(x ,y)--------(x+1, y)
        }
    }
    glEnd();

    //changing the cell count to string values 
    string healthyCellCount = to_string(static_cast<long long>(countHealthy));
    const char* _healthy = healthyCellCount.c_str();
    string cancerCellCount = to_string(static_cast<long long>(countCancer));
    const char* _cancer = cancerCellCount.c_str();
    string medicineCellCount = to_string(static_cast<long long>(countMedicine));
    const char* _medicine = medicineCellCount.c_str();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glColor3f(0, 0, 0); //black

    //for display each cell according to status 
    renderBitmapString(0, 30, opengl_font, "Healthy: ");
    renderBitmapString(0, 50, opengl_font, _healthy);
    renderBitmapString(0, 100, opengl_font, "Cancer: ");
    renderBitmapString(0, 120, opengl_font, _cancer);
    renderBitmapString(0, 170, opengl_font, "Medicine: ");
    renderBitmapString(0, 190, opengl_font, _medicine);
    glPopMatrix();
    glutSwapBuffers();
}


//OpenGL initalization calls
void Init()
{
    glMatrixMode(GL_PROJECTION);
    glViewport(0, 0, ROWS, COLUMNS);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    GLfloat aspect = (GLfloat)ROWS / COLUMNS;
    gluPerspective(45, aspect, 0.1f, 10.0f);
    glClearColor(0.0, 0.0, 0.0, 0.0); 
}




int main(int argc, char** argv)
{
    bool found = false;

    //GPU
    returnCall = clGetPlatformIDs(0, NULL, &gpu_num_of_platforms);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Unable to get platform_id\n");

        return 1;
    }

    cl_platform_id *gpu_platform_ids = new cl_platform_id[gpu_num_of_platforms];

    returnCall = clGetPlatformIDs(gpu_num_of_platforms, gpu_platform_ids, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Unable to get platform_id\n");

        return 1;
    }

    for (int i = 0; i < gpu_num_of_platforms; i++) 
    {

        returnCall = clGetDeviceIDs(gpu_platform_ids[i], CL_DEVICE_TYPE_GPU, 1, &gpu_device_id, &gpu_num_of_devices);

        if (returnCall == CL_SUCCESS) 
        {
            found = true;

            break;
        }
        if (!found) 
        {
            printf("Unable to get device_id\n");

            return 1;
        }
    }

   
    gpu_context = clCreateContext(0, 1, &gpu_device_id, NULL, NULL, &returnCall);

    if (!gpu_context) 
    {
        printf("Error creating GPU context\n");

        return EXIT_FAILURE;
    }

    gpu_command_queue = clCreateCommandQueueWithProperties(gpu_context, gpu_device_id, 0, &returnCall);

    if (!gpu_command_queue) 
    {
        printf("Error creating GPU command queue.\n");

        return EXIT_FAILURE;
    }


    gpu_program = clCreateProgramWithSource(gpu_context, 1, (const char**)&SourceGPUKernel, NULL, &returnCall);

    if (!gpu_program) 
    {
        printf("Error creating GPU program source.\n");

        return EXIT_FAILURE;
    }

    returnCall = clBuildProgram(gpu_program, 0, NULL, NULL, NULL, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        size_t _len;

        // max length of the string memory allocation/ huge array size to stroe maz string legth
        char bigBuffer[2048];

        printf("Error bulding GPU executable.\n");

        clGetProgramBuildInfo(gpu_program, gpu_device_id, CL_PROGRAM_BUILD_LOG, sizeof(bigBuffer), bigBuffer, &_len);

        printf("%s\n", bigBuffer);

        exit(1);
    }

    gpu_kernel = clCreateKernel(gpu_program, "UpdateCellsWithGPU", &returnCall);

    if (!gpu_kernel || returnCall != CL_SUCCESS)
    {
        printf("Error creating kernel.\n");

        exit(1);
    }

    readCell = clCreateBuffer(gpu_context, CL_MEM_READ_ONLY, sizeof(int) * opengl_totalSize, NULL, NULL);
    writeCell = clCreateBuffer(gpu_context, CL_MEM_WRITE_ONLY, sizeof(int) * opengl_totalSize, NULL, NULL);

    if (!readCell || !writeCell) {
        printf("Error to allocate memory to source array.");

        exit(1);
    }



    //CPU 
    returnCall = clGetPlatformIDs(0, NULL, &cpu_num_of_platforms);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Unable to get platform_id\n");

        return 1;
    }

    cl_platform_id* cpu_platform_ids = new cl_platform_id[cpu_num_of_platforms];

    returnCall = clGetPlatformIDs(cpu_num_of_platforms, cpu_platform_ids, NULL);

    if (returnCall != CL_SUCCESS) 
    {
        printf("Unable to get platform_id\n");

        return 1;
    }

    for (int i = 0; i < cpu_num_of_platforms; i++) 
    {
        returnCall = clGetDeviceIDs(cpu_platform_ids[i], CL_DEVICE_TYPE_CPU, 1, &cpu_device_id, &cpu_num_of_devices);

        if (returnCall == CL_SUCCESS)
        {
            found = true;

            break;
        }
        if (!found) 
        {
            printf("Unable to get device_id\n");

            return 1;
        }
    }


    cpu_context = clCreateContext(0, 1, &cpu_device_id, NULL, NULL, &returnCall);

    if (!cpu_context) {

        printf("Error creating CPU context.\n");

        return EXIT_FAILURE;
    }

    cpu_command_queue = clCreateCommandQueueWithProperties(cpu_context, cpu_device_id, 0, &returnCall);

    if (!cpu_command_queue) 
    {
        printf("Error creaing CPU command queueu.\n");

        return EXIT_FAILURE;
    }


    cpu_program = clCreateProgramWithSource(cpu_context, 1, (const char**)&SourceCPUKernel, NULL, &returnCall);

    if (!cpu_program) 
    {
        printf("Error creating host progra.\n");

        return EXIT_FAILURE;
    }


    returnCall = clBuildProgram(cpu_program, 0, NULL, NULL, NULL, NULL);

    if (returnCall != CL_SUCCESS)
    {
        size_t _len;

        char buffer[2048];

        printf("Error creating CPU executable.\n");

        clGetProgramBuildInfo(cpu_program, cpu_device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &_len);

        printf("%s\n", buffer);

        exit(1);
    }

    
    cpu_kernel = clCreateKernel(cpu_program, "UpdateCellsWithCPU", &returnCall);

    if (!cpu_kernel || returnCall != CL_SUCCESS) 
    {
        printf("Error creating CPU kernel.\n");

        exit(1);
    }

  
    colorCell = clCreateBuffer(cpu_context, CL_MEM_READ_ONLY, sizeof(int) * opengl_totalSize, NULL, NULL);

    if (!colorCell) 
    {
        printf("Error allocating device memory.\n");

        exit(1);
    }

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(ROWS, COLUMNS);
    glutCreateWindow("2D Cell Growth Simulation with OpenCL");


    //Initialization of all the cells 
    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLUMNS ; j++)
        {
            //all cells healthy from start
            a_cell[i][j] = HEALTHY;
        }
    }

    srand((int)time(NULL)); //ranadom start time


    //cancer cells randomly scattered but <= 25% 
    for (int i = 0; i <= opengl_cancerInit; i++)
    {
        int x = rand() % ROWS;
        int y = rand() % COLUMNS;

        if (a_cell[x][y] == CANCER)
        {
            i--;
        }
        else
        {
            a_cell[x][y] = CANCER;
        }
    }

    //OpenGL function calls
    glutDisplayFunc(Display);
    glutIdleFunc(Display);
    glutMouseFunc(onClick);
    glutTimerFunc(0, UpdateCells, 0);

    Init();

    glutMainLoop();



    //cleaning

    //source array
    clReleaseMemObject(readCell);
    clReleaseMemObject(writeCell);
    //programs
    clReleaseProgram(gpu_program);
    clReleaseProgram(cpu_program);
    //kernels
    clReleaseKernel(gpu_kernel);
    clReleaseKernel(cpu_kernel);
    //command queues 
    clReleaseCommandQueue(gpu_command_queue);
    clReleaseCommandQueue(cpu_command_queue);
    //contexts
    clReleaseContext(gpu_context);
    clReleaseContext(cpu_context);

    return 0;
}
