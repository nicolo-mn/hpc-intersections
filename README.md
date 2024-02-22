<h1 align="center">⏺️HPC Intersections Algorithm</h1>

<p align="left">
This project is meant to parallelize a program with quadratic asyntotic cost using Open-MP and either MPI or CUDA, which were both explored during the course. The purpose of the algorithm is to move some circles that are overlapping to avoid collisions. The scalability and efficiency of the solutions were measured through different metrics. The calculations, as well as explanations of the parallelization techniques used, can be found in the report (Relazione.pdf). This is a project made for the <a href="https://www.unibo.it/en/study/phd-professional-masters-specialisation-schools-and-other-programmes/course-unit-catalogue/course-unit/2023/385080">HPC course</a> of the Computer Science and Engineering Bachelor's Degree at the University of Bologna.
</p>

<p align="left">
To compile use either make cuda-circles or make omp-circles. <br>
To execute use ./cuda-circles [ncircles [iterations]] or ./omp-circles [ncircles [iterations]].
</p>

