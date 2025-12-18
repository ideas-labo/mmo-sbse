package mmo;

import jmetal.core.SolutionSet;
import jmetal.util.Distance;
import jmetal.util.PseudoRandom;
import jmetal.util.comparators.CrowdingComparator;
import jmetal.util.ranking.NondominatedRanking;
import spl.MAP_test;
import spl.fm.Product;
import spl.techniques.QD.IndividualMultiObj;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class NSGA2Optimizer {
    public static final int FITNESS_TYPE_ACCURATE = 0;
    public static final int FITNESS_TYPE_APPORIMATE = 1;
    public static final int FITNESS_TYPE_DIVERSITY = 2;
    public static final File workingDir = new File(System.getProperty("user.dir"));
    private static final String INITIAL_POPULATION_DIR = workingDir + "/../../Results/initial_populations/NSGA2/";
    private String initialPopulationFilePath;

    public static final int MODE_FT_FA = 0;
    public static final int MODE_G1_G2 = 1;
    public static final int MODE_AGE = 2;
    public static final int MODE_GAUSSIAN = 3;
    public static final int MODE_RECIPROCAL = 4;
    public static final int MODE_PENALTY = 5;
    public static final int MODE_NOVELTY = 6;
    public static final int MODE_DIVERSITY = 7;

    private static class MultiProcessConfig {
        int maxCpuCores = 50;
        boolean enableMultiProcess = true;
        long taskTimeoutHours = 24;
        boolean useProcessIsolation = true;
        long maxMemoryPerProcessMB = 1024;

        public MultiProcessConfig() {}

        public MultiProcessConfig(int maxCpuCores, boolean enableMultiProcess, long taskTimeoutHours) {
            this.maxCpuCores = maxCpuCores;
            this.enableMultiProcess = enableMultiProcess;
            this.taskTimeoutHours = taskTimeoutHours;
            this.useProcessIsolation = true;
            this.maxMemoryPerProcessMB = 2048;
        }
    }

    private static class TaskInfo {
        final String dataset;
        final String fmFile;
        final int run;
        final long seedValue;
        final int optimizationMode;
        final List<Product> seed;

        public TaskInfo(String dataset, String fmFile, int run, long seedValue,
                        int optimizationMode, List<Product> seed) {
            this.dataset = dataset;
            this.fmFile = fmFile;
            this.run = run;
            this.seedValue = seedValue;
            this.optimizationMode = optimizationMode;
            this.seed = seed;
        }

        @Override
        public String toString() {
            return String.format("%s_run%d_mode%d_seed%d", dataset, run, optimizationMode, seedValue);
        }
    }

    private static class TaskResult {
        final TaskInfo taskInfo;
        final boolean success;
        final long executionTimeMs;
        final String errorMessage;
        final Map<String, Object> result;

        public TaskResult(TaskInfo taskInfo, boolean success, long executionTimeMs,
                          String errorMessage, Map<String, Object> result) {
            this.taskInfo = taskInfo;
            this.success = success;
            this.executionTimeMs = executionTimeMs;
            this.errorMessage = errorMessage;
            this.result = result;
        }
    }

    private SolutionSet population;
    private int cellsInEachDim;
    private int lowerBound = 1;
    private int upperBound = 100;
    private long timeAllowedMS;
    private long evaluations = 0;
    private int initialCriticalPoint;
    private int optimizationMode;
    private Map<UUID, Integer> ageInfo;
    private String currentFilePath;
    private int currentGeneration = 1;
    private int maxGenerations;
    private List<GenerateFA.ArchiveEntry> noveltyArchive = new ArrayList<>();
    private int maxArchiveSize = 1000;
    private Random random;
    private boolean useSeed;
    private int RandomSeed;

    private List<Double> pValuesHistory = new ArrayList<>();
    int bestGeneration = 0;
    double bestOriginalObjective1 = Double.POSITIVE_INFINITY;
    private double bestOriginalObjective2 = Double.POSITIVE_INFINITY;
    private String csvOutputPath;
    private long startTime;
    private String datasetName;
    private String modeStr;

    public NSGA2Optimizer(int cellsInEachDim, long timeAllowedMS, int optimizationMode, String filePath, long seed, boolean useSeed) {
        this.cellsInEachDim = cellsInEachDim;
        this.timeAllowedMS = timeAllowedMS;
        this.optimizationMode = optimizationMode;
        this.currentFilePath = filePath;
        this.random = new Random(seed);
        this.useSeed = useSeed;
        initializeAgeInfo();
        initializePopulationFile(filePath, seed);
    }

    public NSGA2Optimizer(int lowerBound, int upperBound, long timeAllowedMS, int optimizationMode, String filePath, long seed, boolean useSeed) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
        this.cellsInEachDim = upperBound - lowerBound + 1;
        this.timeAllowedMS = timeAllowedMS;
        this.optimizationMode = optimizationMode;
        this.currentFilePath = filePath;
        this.random = new Random(seed);
        this.useSeed = useSeed;
        initializeAgeInfo();
        initializePopulationFile(filePath, seed);
    }

    private void initializePopulationFile(String filePath, long seed) {
        String problemName = new File(filePath).getName().replace(".dimacs", "");
        String modeName = getModeString(optimizationMode);
        String fileName = String.format("%s_%s_%d.pop", problemName, modeName, seed);
        new File(INITIAL_POPULATION_DIR).mkdirs();
        this.initialPopulationFilePath = INITIAL_POPULATION_DIR + fileName;
    }

    public void runNSGA2(List<Product> seed, long seedValue) throws Exception {
        startTime = System.currentTimeMillis();
        Distance distance = new Distance();
        int fitnessType = FITNESS_TYPE_ACCURATE;

        if (fitnessType == FITNESS_TYPE_DIVERSITY) {
            IndividualMultiObj.useDistanceAsFitness = true;
        }
        evaluations = 0;
        IndividualMultiObj.counter = 0;
        int populationSize = cellsInEachDim;

        population = initializePopulation(seed, populationSize, fitnessType, seedValue);
        evaluations += populationSize;

        System.out.println("\n========== Initial Population Best Solution ==========");
        IndividualMultiObj initialBest = findBestSolution(population);
        double initialCoverage = Math.abs(initialBest.getOriginalObjectives()[0]);
        if (initialCoverage > 1.0) {
            initialCoverage /= 100.0;
        }
        System.out.printf("Size: %d, Coverage: %.2f%%%n",
                initialBest.getSize(), initialCoverage * 100);

        System.out.println("Initial upperBound = " + upperBound);
        for (int i = 0; i < upperBound - lowerBound + 1; i++) {
            if (i < population.size()) {
                population.get(i).setMaxProductsNo(upperBound);
            }
        }

        if (optimizationMode == MODE_NOVELTY) {
            noveltyArchive.clear();
            System.out.println("Novelty archive initialized empty list");
            List<IndividualMultiObj> initialPop = population.getSolutionsList().stream()
                    .map(s -> (IndividualMultiObj) s)
                    .collect(Collectors.toList());
            GenerateFA.updateNoveltyArchive(initialPop, noveltyArchive, maxArchiveSize);
            System.out.println("Novelty archive updated after initial population evaluation, size: " + noveltyArchive.size());
        }

        String problem = MAP_test.getInstance().getDimacsFile();
        datasetName = new File(problem).getName().replace(".dimacs", "");
        modeStr = getFullModeName(optimizationMode);

        String csvFileName = datasetName + "-" + seedValue + "_" + modeStr + ".csv";

        this.csvOutputPath = workingDir+ "/../../../../Results/RQ1-raw-data/SPLT/" +csvFileName;

        initializeCSVFile();

        handlePopulationObjectives(population);

        List<IndividualMultiObj> offspringPopulation = new ArrayList<>(populationSize);
        NondominatedRanking ranking = null;

        while (evaluations < 12000 && (System.currentTimeMillis() - startTime) < 24 * 60 * 60 * 1000) {

            offspringPopulation.clear();

            for (int i = 0; i < populationSize; i++) {
                int ind = PseudoRandom.randInt(0, upperBound - lowerBound);
                IndividualMultiObj selected = (IndividualMultiObj) population.get(ind);
                IndividualMultiObj mutated = new IndividualMultiObj(selected);
                mutated.mutate();
                mutated.fitness(fitnessType);

                if (optimizationMode == MODE_AGE) {
                    mutated.setUuid(UUID.randomUUID());
                    ageInfo.put(mutated.getUuid(), populationSize + currentGeneration);
                }

                mutated.setOriginalObjectives(new double[]{
                        mutated.getObjective(0),
                        mutated.getObjective(1)
                });
                evaluations++;
                offspringPopulation.add(mutated);
            }

            if (optimizationMode == MODE_NOVELTY) {
                GenerateFA.updateNoveltyArchive(offspringPopulation, noveltyArchive, maxArchiveSize);
            }

            SolutionSet union_ = population.union(offspringPopulation);

            double pValue = calculatePValue(union_);
            pValuesHistory.add(pValue);

            handlePopulationObjectives(union_);

            boolean useDuplicateHandling = false;
            switch (optimizationMode) {
                case MODE_AGE:
                case MODE_GAUSSIAN:
                case MODE_RECIPROCAL:
                case MODE_G1_G2:
                case MODE_PENALTY:
                case MODE_NOVELTY:
                case MODE_DIVERSITY:
                    useDuplicateHandling = true;
                    break;
                case MODE_FT_FA:
                default:
                    useDuplicateHandling = false;
                    break;
            }

            ranking = new NondominatedRanking(union_, useDuplicateHandling);

            int remain = populationSize;
            int index = 0;
            SolutionSet front;
            population.clear();

            front = ranking.getSubfront(index);
            while ((remain > 0) && (remain >= front.size())) {
                distance.crowdingDistanceAssignment(front, 2);
                for (int k = 0; k < front.size(); k++) {
                    population.add(front.get(k));
                }
                remain -= front.size();
                index++;
                if (remain > 0) {
                    front = ranking.getSubfront(index);
                }
            }

            if (remain > 0) {
                distance.crowdingDistanceAssignment(front, 2);
                front.sort(new CrowdingComparator());
                for (int k = 0; k < remain; k++) {
                    population.add(front.get(k));
                }
            }

            if (optimizationMode == MODE_AGE) {
                Set<UUID> currentUuids = new HashSet<>();
                for (int i = 0; i < population.size(); i++) {
                    IndividualMultiObj indiv = (IndividualMultiObj) population.get(i);
                    currentUuids.add(indiv.getUuid());
                }
                ageInfo.keySet().removeIf(uuid -> !currentUuids.contains(uuid));
            }

            updateBestSolution(population, currentGeneration);
            System.out.printf("%s %s | Gen %d | Evaluations: %d/12000 | Time: %.1fs | Best: %.2f\n",
                    datasetName, modeStr, currentGeneration, evaluations,
                    (System.currentTimeMillis() - startTime) / 1000.0,
                    -bestOriginalObjective1);
            writeGenerationInfoToCSV(currentGeneration, pValue, -bestOriginalObjective1);
            currentGeneration++;
        }

        writeFinalResultsToCSV();
    }

    private static TaskResult executeSingleTaskWithProcessIsolation(TaskInfo taskInfo, boolean useSeed,
                                                                    int nbProds, String samplingMethod, String outputDir,
                                                                    MultiProcessConfig mpConfig) {
        long startTime = System.currentTimeMillis();

        ProcessBuilder processBuilder = new ProcessBuilder();


        List<String> command = new ArrayList<>();

        long processTimeoutMs = Math.max(0L, mpConfig.taskTimeoutHours * 60L * 60L * 1000L);
        final long GRACE_PERIOD_MS = 60L * 60L * 1000L;

        command.add("java");
        command.add("-Dnsga2.process.timeout.ms=" + processTimeoutMs);
        command.add("-Dnsga2.process.grace.ms=" + GRACE_PERIOD_MS);

        command.add("-Xmx" + mpConfig.maxMemoryPerProcessMB + "m");
        command.add("-XX:+UseG1GC");
        command.add("-XX:MaxGCPauseMillis=200");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add("mmo.NSGA2ProcessExecutor");
        command.add(taskInfo.dataset);
        command.add(taskInfo.fmFile);
        command.add(String.valueOf(taskInfo.run));
        command.add(String.valueOf(taskInfo.seedValue));
        command.add(String.valueOf(taskInfo.optimizationMode));
        command.add(String.valueOf(useSeed));
        command.add(String.valueOf(nbProds));
        command.add(samplingMethod);
        command.add(outputDir);

        processBuilder.command(command);
        processBuilder.redirectErrorStream(true);

        Process process = null;
        StringBuilder output = new StringBuilder();

        try {
            System.out.printf("[NSGA2 Process Isolation] Starting isolated process for: %s%n", taskInfo);

            process = processBuilder.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;

            long processEndTime = startTime + processTimeoutMs;
            boolean processCompleted = false;

            while (System.currentTimeMillis() < processEndTime && !processCompleted) {
                try {
                    int exitCode = process.exitValue();
                    processCompleted = true;

                    while ((line = reader.readLine()) != null) {
                        output.append(line).append("\n");
                        System.out.printf("[NSGA2-Process-%s] %s%n", taskInfo, line);
                    }
                    break;
                } catch (IllegalThreadStateException e) {
                    try {
                        if (reader.ready()) {
                            line = reader.readLine();
                            if (line != null) {
                                output.append(line).append("\n");
                                System.out.printf("[NSGA2-Process-%s] %s%n", taskInfo, line);
                            }
                        } else {
                            Thread.sleep(100);
                        }
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (IOException ioex) {
                    }
                }
            }

            long executionTime = System.currentTimeMillis() - startTime;

            if (System.currentTimeMillis() >= processEndTime && !processCompleted) {
                System.out.printf("[NSGA2 Process Isolation] Process reached configured timeout for: %s, attempting graceful termination (grace=%d ms)%n",
                        taskInfo, GRACE_PERIOD_MS);

                try {
                    process.destroy();

                    long waitStart = System.currentTimeMillis();
                    boolean exitedGracefully = false;

                    while (System.currentTimeMillis() - waitStart < GRACE_PERIOD_MS) {
                        try {
                            if (process.waitFor(500, TimeUnit.MILLISECONDS)) {
                                exitedGracefully = true;
                                break;
                            }
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                            break;
                        }

                        try {
                            while (reader.ready() && (line = reader.readLine()) != null) {
                                output.append(line).append("\n");
                                System.out.printf("[NSGA2-Process-%s] %s%n", taskInfo, line);
                            }
                        } catch (IOException ioe) {
                        }
                    }

                    try {
                        while ((line = reader.readLine()) != null) {
                            output.append(line).append("\n");
                            System.out.printf("[NSGA2-Process-%s] %s%n", taskInfo, line);
                        }
                    } catch (IOException ioe) {
                    }

                    long totalExecution = System.currentTimeMillis() - startTime;

                    if (!exitedGracefully) {
                        System.out.printf("[NSGA2 Process Isolation] Grace period expired for: %s; forcing termination now%n", taskInfo);
                        process.destroyForcibly();
                        try {
                            process.waitFor(5, TimeUnit.SECONDS);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                        }
                    } else {
                        System.out.printf("[NSGA2 Process Isolation] Child exited gracefully during grace period for: %s%n", taskInfo);
                    }

                    Map<String, Object> timeoutResult = parseProcessOutput(output.toString());
                    if (!timeoutResult.containsKey("terminationReason")) {
                        timeoutResult.put("terminationReason", "time_limit_reached");
                    }
                    if (!timeoutResult.containsKey("bestFT")) {
                        timeoutResult.put("bestFT", 0.0);
                    }
                    if (!timeoutResult.containsKey("budgetUsed")) {
                        timeoutResult.put("budgetUsed", 0);
                    }

                    return new TaskResult(taskInfo, true, totalExecution,
                            "Normal termination: process-timeout reached (graceful wait)", timeoutResult);

                } catch (Exception e) {
                    long execTime = System.currentTimeMillis() - startTime;
                    String errorMsg = "Failed during graceful termination: " + e.getMessage();
                    System.err.printf("[NSGA2 Process Isolation] Graceful termination failed for %s: %s%n", taskInfo, errorMsg);

                    if (process != null) {
                        try {
                            process.destroyForcibly();
                        } catch (Exception ex) {
                        }
                    }

                    Map<String, Object> fallbackResult = parseProcessOutput(output.toString());
                    fallbackResult.putIfAbsent("terminationReason", "time_limit_reached");
                    fallbackResult.putIfAbsent("bestFT", 0.0);
                    fallbackResult.putIfAbsent("budgetUsed", 0);

                    return new TaskResult(taskInfo, true, execTime, errorMsg, fallbackResult);
                }
            }

            if (processCompleted) {
                int exitCode = process.exitValue();

                if (exitCode == 0) {
                    System.out.printf("[NSGA2 Process Isolation] Task completed successfully: %s, Time: %.2f seconds%n",
                            taskInfo, executionTime / 1000.0);

                    Map<String, Object> result = parseProcessOutput(output.toString());
                    return new TaskResult(taskInfo, true, executionTime, null, result);
                } else {
                    String outputStr = output.toString();
                    if (outputStr.contains("time_limit_reached") || outputStr.contains("24-hour") ||
                            outputStr.contains("Normal termination")) {
                        System.out.printf("[NSGA2 Process Isolation] Task normal termination (timeout or normal): %s, Time: %.2f seconds%n",
                                taskInfo, executionTime / 1000.0);

                        Map<String, Object> result = parseProcessOutput(outputStr);
                        if (!result.containsKey("terminationReason")) {
                            result.put("terminationReason", "time_limit_reached");
                        }
                        return new TaskResult(taskInfo, true, executionTime, "Normal timeout/termination", result);
                    }

                    String errorMsg = String.format("Process exited with code %d. Output: %s", exitCode, outputStr);
                    System.err.printf("[NSGA2 Process Isolation] Task failed: %s, Error: %s%n", taskInfo, errorMsg);
                    return new TaskResult(taskInfo, false, executionTime, errorMsg, null);
                }
            }

        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Process execution failed: " + e.getMessage();
            System.err.printf("[NSGA2 Process Isolation] Task failed: %s, Error: %s%n", taskInfo, errorMsg);

            if (process != null) {
                process.destroyForcibly();
            }

            return new TaskResult(taskInfo, false, executionTime, errorMsg, null);
        } finally {
            if (process != null) {
                try {
                    process.getInputStream().close();
                } catch (IOException ignored) {}
                try {
                    process.getOutputStream().close();
                } catch (IOException ignored) {}
                try {
                    process.getErrorStream().close();
                } catch (IOException ignored) {}
            }
        }

        long executionTime = System.currentTimeMillis() - startTime;
        return new TaskResult(taskInfo, false, executionTime, "Unexpected execution path", null);
    }

    private static Map<String, Object> parseProcessOutput(String output) {
        Map<String, Object> result = new HashMap<>();
        try {
            String[] lines = output.split("\n");
            for (String line : lines) {
                line = line.trim();
                if (line.contains("Best Solution: 'ft':")) {
                    try {
                        String[] parts = line.split("'ft':");
                        if (parts.length > 1) {
                            String ftValue = parts[1].split(",")[0].trim();
                            result.put("bestFT", Double.parseDouble(ftValue));
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Failed to parse BestFt from line: " + line);
                    }
                } else if (line.contains("budget_used:")) {
                    try {
                        String[] parts = line.split("budget_used:");
                        if (parts.length > 1) {
                            String budgetStr = parts[1].trim();
                            result.put("budgetUsed", Integer.parseInt(budgetStr));
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Failed to parse budget_used from line: " + line);
                    }
                } else if (line.contains("Termination Reason:")) {
                    String[] parts = line.split("Termination Reason:");
                    if (parts.length > 1) {
                        result.put("terminationReason", parts[1].trim());
                    }
                } else if (line.contains("NSGA-II completed") && line.contains("Budget used=")) {
                    try {
                        if (line.contains("Budget used=")) {
                            String budgetPart = line.substring(line.indexOf("Budget used=") + 12);
                            String budgetStr = budgetPart.split("/")[0].trim();
                            result.put("budgetUsed", Integer.parseInt(budgetStr));
                        }
                        if (line.contains("Best ft=")) {
                            String ftPart = line.substring(line.indexOf("Best ft=") + 8);
                            String ftStr = ftPart.split(",")[0].trim();
                            if (!ftStr.equals("Infinity") && !ftStr.equals("∞")) {
                                result.put("bestFT", Double.parseDouble(ftStr));
                            }
                        }
                    } catch (Exception e) {
                        System.err.println("Failed to parse completion line: " + line);
                    }
                }
            }

            if (!result.containsKey("terminationReason")) {
                if (output.contains("time_limit_reached") || output.contains("24-hour") ||
                        output.contains("timeout")) {
                    result.put("terminationReason", "time_limit_reached");
                } else if (output.contains("budget_exhausted")) {
                    result.put("terminationReason", "budget_exhausted");
                } else {
                    result.put("terminationReason", "unknown");
                }
            }

            if (!result.containsKey("bestFT")) {
                result.put("bestFT", 0.0);
            }

            if (!result.containsKey("budgetUsed")) {
                result.put("budgetUsed", 0);
            }

        } catch (Exception e) {
            System.err.println("Error parsing process output: " + e.getMessage());
            result.clear();
            result.put("terminationReason", "parse_error");
            result.put("bestFT", 0.0);
            result.put("budgetUsed", 0);
        }
        return result;
    }

    private static void executeTasksInParallelWithIsolation(List<TaskInfo> tasks, boolean useSeed,
                                                            int nbProds, String samplingMethod, String outputDir,
                                                            MultiProcessConfig mpConfig) throws InterruptedException {

        BlockingQueue<Runnable> workQueue = new LinkedBlockingQueue<>(tasks.size());
        ExecutorService executor = new ThreadPoolExecutor(
                mpConfig.maxCpuCores,
                mpConfig.maxCpuCores,
                60L, TimeUnit.SECONDS,
                workQueue,
                new ThreadFactory() {
                    private final AtomicInteger threadNumber = new AtomicInteger(1);
                    @Override
                    public Thread newThread(Runnable r) {
                        Thread thread = new Thread(r, "NSGA2-Process-Worker-" + threadNumber.getAndIncrement());
                        thread.setDaemon(true);
                        return thread;
                    }
                },
                new ThreadPoolExecutor.CallerRunsPolicy()
        );

        CompletionService<TaskResult> completionService = new ExecutorCompletionService<>(executor);

        List<Future<TaskResult>> futures = new ArrayList<>();
        for (TaskInfo task : tasks) {
            Future<TaskResult> future = completionService.submit(() -> {
                if (mpConfig.useProcessIsolation) {
                    return executeSingleTaskWithProcessIsolation(task, useSeed, nbProds, samplingMethod, outputDir, mpConfig);
                } else {
                    return executeSingleTaskInThread(task, useSeed, nbProds, samplingMethod, outputDir);
                }
            });
            futures.add(future);
        }

        long overallStartTime = System.currentTimeMillis();
        int completed = 0;
        int failed = 0;

        try {
            for (int i = 0; i < tasks.size(); i++) {
                try {
                    Future<TaskResult> future = completionService.poll(mpConfig.taskTimeoutHours, TimeUnit.HOURS);

                    if (future == null) {
                        System.out.printf("Overall execution time limit reached after %d hours, waiting for remaining tasks...%n", mpConfig.taskTimeoutHours);
                        break;
                    }

                    TaskResult result = future.get();

                    if (result.success) {
                        completed++;
                        String terminationReason = result.result != null ?
                                (String)result.result.get("terminationReason") : "unknown";

                        System.out.printf("[NSGA2 Main Process] Task completed: %s (Time: %.1f seconds, Termination: %s)%n",
                                result.taskInfo, result.executionTimeMs / 1000.0, terminationReason);
                    } else {
                        failed++;
                        System.err.printf("[NSGA2 Main Process] Task failed: %s, Error: %s%n",
                                result.taskInfo, result.errorMessage);
                    }

                    if ((completed + failed) % Math.max(1, tasks.size() / 10) == 0) {
                        long elapsedTime = System.currentTimeMillis() - overallStartTime;
                        double avgTimePerTask = elapsedTime / (double)(completed + failed);
                        long estimatedRemaining = (long)(avgTimePerTask * (tasks.size() - completed - failed));

                        System.out.printf("[NSGA2 Main Process] Progress: %d/%d completed, %d failed, Elapsed: %.1f minutes, Estimated: %.1f minutes%n",
                                completed, tasks.size(), failed,
                                elapsedTime / 60000.0, estimatedRemaining / 60000.0);
                    }

                } catch (ExecutionException e) {
                    failed++;
                    Throwable cause = e.getCause();
                    String errorMsg = cause != null ? cause.getMessage() : e.getMessage();
                    System.err.printf("[NSGA2 Main Process] Task execution exception: %s%n", errorMsg);
                } catch (CancellationException e) {
                    failed++;
                    System.err.printf("[NSGA2 Main Process] Task cancelled%n");
                }
            }
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                    System.out.println("[NSGA2 Main Process] Some tasks are still running, forcing shutdown...");
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        long totalTime = System.currentTimeMillis() - overallStartTime;
        System.out.printf("%nNSGA2 Parallel execution completed: %d/%d succeeded, %d failed, Total time: %.2f minutes%n",
                completed, tasks.size(), failed, totalTime / 60000.0);
    }

    private static TaskResult executeSingleTaskInThread(TaskInfo taskInfo, boolean useSeed,
                                                        int nbProds, String samplingMethod, String outputDir) {
        long startTime = System.currentTimeMillis();
        long timeoutMs = 24 * 60 * 60 * 1000;

        NSGA2Optimizer optimizer = null;

        try {
            System.out.printf("[NSGA2 Thread] Starting task execution: %s%n", taskInfo);

            synchronized (MAP_test.class) {
                MAP_test.getInstance().initializeModelSolvers(taskInfo.fmFile, 2);
            }

            List<Product> seed = taskInfo.seed;
            if (useSeed && (seed == null || seed.isEmpty())) {
                String fmFileName = new File(taskInfo.fmFile).getName();
                String seedPath = outputDir + "SAT4J/" + fmFileName + "/Samples/" + nbProds + "prods/Products.0";
                File seedFile = new File(seedPath);

                if (!seedFile.exists()) {
                    System.out.printf("[NSGA2 Thread] %s: Seed file does not exist, generating seed...%n", taskInfo.dataset);
                    synchronized (MAP_test.class) {
                        MAP_test.getInstance().generateSeeds(taskInfo.fmFile, outputDir, 10, nbProds, samplingMethod);
                    }
                }
                synchronized (MAP_test.class) {
                    seed = MAP_test.getInstance().loadSeedsFromFile(seedPath);
                }
                if (seed == null || seed.isEmpty()) {
                    throw new RuntimeException("Seed loading failed or is empty: " + seedPath);
                }
                System.out.printf("[NSGA2 Thread] %s: Loaded seed size: %d%n", taskInfo.dataset, seed.size());
            }

            optimizer = new NSGA2Optimizer(1, 8, 12000, taskInfo.optimizationMode,
                    taskInfo.fmFile, taskInfo.seedValue, useSeed);
            Map<String, Object> result = runNSGA2WithRetry(optimizer, seed, taskInfo.seedValue, 3);

            long executionTime = System.currentTimeMillis() - startTime;
            String terminationReason = (String) result.get("terminationReason");
            System.out.printf("[NSGA2 Thread] Task completed: %s, Time: %.2f seconds, Termination: %s%n",
                    taskInfo, executionTime / 1000.0, terminationReason);

            return new TaskResult(taskInfo, true, executionTime, null, result);

        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Task execution failed: " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            System.err.printf("[NSGA2 Thread] Task failed: %s, Error: %s%n", taskInfo, errorMsg);

            if (executionTime > timeoutMs) {
                System.out.printf("[NSGA2 Thread] Task normal termination (timeout): %s, Time: %.2f seconds%n",
                        taskInfo, executionTime / 1000.0);

                Map<String, Object> timeoutResult = new HashMap<>();
                timeoutResult.put("bestFT", 0.0);
                timeoutResult.put("budgetUsed", 0);
                timeoutResult.put("terminationReason", "time_limit_reached");

                return new TaskResult(taskInfo, true, executionTime, "Normal timeout termination", timeoutResult);
            }

            String fullErrorMsg = String.format("%s: %s", e.getClass().getSimpleName(), errorMsg);
            if (e.getCause() != null) {
                fullErrorMsg += " - Cause: " + e.getCause().getMessage();
            }

            return new TaskResult(taskInfo, false, executionTime, fullErrorMsg, null);
        } finally {
            if (optimizer != null) {
                try {
                    optimizer.cleanupMemory();
                } catch (Exception cleanupEx) {
                    System.err.println("Failed to cleanup optimizer: " + cleanupEx.getMessage());
                }
            }
        }
    }

    private static Map<String, Object> runNSGA2WithRetry(NSGA2Optimizer optimizer, List<Product> seed,
                                                         long seedValue, int maxRetries) throws Exception {
        Exception lastException = null;

        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                optimizer.runNSGA2(seed, seedValue);

                Map<String, Object> result = new HashMap<>();
                result.put("bestFT", -optimizer.bestOriginalObjective1);
                result.put("budgetUsed", optimizer.evaluations);
                result.put("terminationReason", "completed");
                result.put("bestGeneration", optimizer.bestGeneration);

                return result;
            } catch (Exception e) {
                lastException = e;
                System.err.printf("[NSGA2 Retry] Execution failed, retrying for the %d-th time, Error: %s%n", attempt, e.getMessage());

                if (attempt < maxRetries) {
                    Thread.sleep(1000 * attempt);
                    synchronized (MAP_test.class) {
                        MAP_test.getInstance().initializeModelSolvers(optimizer.currentFilePath, 2);
                    }
                }
            }
        }

        throw new Exception("NSGA2 execution failed after " + maxRetries + " retries", lastException);
    }

    public static void runMultiProcessNSGA2(String[] datasets, String basePath, String outputDir,
                                            int runs, int nbProds, String samplingMethod,
                                            boolean useSeed, int[] optimizationModes,
                                            MultiProcessConfig mpConfig) throws Exception {

        System.out.printf("=== Starting Multi-process NSGA2 Optimization Tasks ===%n");
        System.out.printf("Number of datasets: %d, Number of runs per dataset: %d, Modes: %d%n",
                datasets.length, runs, optimizationModes.length);
        System.out.printf("Total tasks: %d, Budget per task: %d, Population size: %d%n",
                datasets.length * runs * optimizationModes.length, 12000, 8);
        System.out.printf("Multi-process: %s, Process isolation: %s, CPU cores: %d%n",
                mpConfig.enableMultiProcess ? "Enabled" : "Disabled",
                mpConfig.useProcessIsolation ? "Enabled" : "Disabled",
                mpConfig.maxCpuCores);
        System.out.printf("Time limit: %d hours, Memory limit: %d MB per process%n",
                mpConfig.taskTimeoutHours, mpConfig.maxMemoryPerProcessMB);
        System.out.printf("Start time: %s%n", new Date());

        List<TaskInfo> allTasks = generateAllTasks(datasets, basePath, outputDir,
                runs, nbProds, samplingMethod, useSeed, optimizationModes);

        if (!mpConfig.enableMultiProcess) {
            executeTasksSequentially(allTasks, useSeed, nbProds, samplingMethod, outputDir);
        } else {
            executeTasksInParallelWithIsolation(allTasks, useSeed, nbProds, samplingMethod, outputDir, mpConfig);
        }

        System.out.printf("%n=== All NSGA2 Tasks Completed ===%n");
        System.out.printf("End time: %s%n", new Date());
    }

    private static List<TaskInfo> generateAllTasks(String[] datasets, String basePath, String outputDir,
                                                   int runs, int nbProds, String samplingMethod,
                                                   boolean useSeed, int[] optimizationModes) {
        List<TaskInfo> tasks = new ArrayList<>();

        for (String dataset : datasets) {
            String fmFile = basePath + dataset + ".dimacs";

            List<Product> seed = null;
            if (useSeed) {
                try {
                    String fmFileName = new File(fmFile).getName();
                    String seedPath = outputDir + "SAT4J/" + fmFileName + "/Samples/" + nbProds + "prods/Products.0";
                    File seedFile = new File(seedPath);

                    if (seedFile.exists()) {
                        seed = Collections.emptyList();
                        System.out.printf("Dataset %s seed file exists, will be loaded in the task%n", dataset);
                    } else {
                        System.out.printf("Dataset %s seed file does not exist, will be generated during runtime%n", dataset);
                        seed = new ArrayList<>();
                    }
                } catch (Exception e) {
                    System.err.printf("Warning: Seed check failed for dataset %s, will be handled during runtime: %s%n", dataset, e.getMessage());
                    seed = new ArrayList<>();
                }
            }

            for (int run = 0; run < runs; run++) {
                for (int optimizationMode : optimizationModes) {
                    long seedValue = run;
                    tasks.add(new TaskInfo(dataset, fmFile, run, seedValue, optimizationMode, seed));
                }
            }
        }

        return tasks;
    }

    private static void executeTasksSequentially(List<TaskInfo> tasks, boolean useSeed,
                                                 int nbProds, String samplingMethod, String outputDir) {
        System.out.printf("Starting sequential execution of %d tasks...%n", tasks.size());

        long overallStartTime = System.currentTimeMillis();
        int completed = 0;
        int failed = 0;

        for (TaskInfo task : tasks) {
            System.out.printf("%n--- Executing Task %d/%d: %s ---%n",
                    completed + failed + 1, tasks.size(), task);

            TaskResult result = executeSingleTaskInThread(task, useSeed, nbProds, samplingMethod, outputDir);

            if (result.success) {
                completed++;
                String terminationReason = result.result != null ?
                        (String)result.result.get("terminationReason") : "unknown";
                System.out.printf("Task completed: %s, Termination: %s%n", task, terminationReason);
            } else {
                failed++;
                System.err.printf("Task failed: %s, Error: %s%n", task, result.errorMessage);
            }

            long elapsedTime = System.currentTimeMillis() - overallStartTime;
            double avgTimePerTask = elapsedTime / (double)(completed + failed);
            long estimatedRemaining = (long)(avgTimePerTask * (tasks.size() - completed - failed));

            System.out.printf("Progress: %d/%d completed, %d failed, Elapsed: %.1f minutes, Estimated: %.1f minutes%n",
                    completed, tasks.size(), failed,
                    elapsedTime / 60000.0, estimatedRemaining / 60000.0);
        }

        long totalTime = System.currentTimeMillis() - overallStartTime;
        System.out.printf("%nSequential execution completed: %d/%d succeeded, %d failed, Total time: %.2f minutes%n",
                completed, tasks.size(), failed, totalTime / 60000.0);
    }

    private IndividualMultiObj findBestSolution(SolutionSet population) {
        IndividualMultiObj best = null;
        double bestFitness = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < population.size(); i++) {
            IndividualMultiObj indiv = (IndividualMultiObj) population.get(i);
            double fitness = -indiv.getObjective(0);
            if (fitness > bestFitness) {
                bestFitness = fitness;
                best = indiv;
            }
        }
        return best;
    }

    private void initializeAgeInfo() {
        if (optimizationMode == MODE_AGE) {
            ageInfo = new HashMap<>();
        }
    }

    private SolutionSet initializePopulation(List<Product> seed, int populationSize, int fitnessType, long seedValue) throws Exception {
        SolutionSet population = new SolutionSet(populationSize);
        boolean reachFirst = false;
        initialCriticalPoint = upperBound;

        String problemName = new File(currentFilePath).getName().replace(".dimacs", "");
        String fileName = String.format("%s_%d.pop", problemName, seedValue);
        String initialPopulationFilePath = INITIAL_POPULATION_DIR + fileName;

        if (loadInitialPopulationFromFile(population, populationSize, initialPopulationFilePath)) {
            System.out.println("Successfully loaded initial population from file: " + initialPopulationFilePath);
            if (optimizationMode == MODE_AGE) {
                for (int i = 0; i < population.size(); i++) {
                    IndividualMultiObj indiv = (IndividualMultiObj) population.get(i);
                    ageInfo.put(indiv.getUuid(), i + 1);
                }
            }
            return population;
        }

        System.out.println("Creating new initial population and saving to: " + initialPopulationFilePath);

        for (int i = 0; i < cellsInEachDim; i++) {
            List<Product> temp = new ArrayList<>();
            if (useSeed && seed != null && !seed.isEmpty() && seed.size() == lowerBound + i) {
                temp = seed;
                System.out.println("**************Seeding with provided seed**************");
            } else {
                int count = 0;
                while (count < lowerBound + i) {
                    temp.add(MAP_test.getInstance().getOneRandomProductSAT4J());
                    count++;
                }
            }

            IndividualMultiObj indiv = createIndividual(temp);
            population.add(i, indiv);

            if (optimizationMode == MODE_AGE) {
                ageInfo.put(indiv.getUuid(), i + 1);
            }

            if (fitnessType == FITNESS_TYPE_ACCURATE) {
                if (!reachFirst && -indiv.getObjective(0) >= 100.0 - 1e-6) {
                    upperBound = lowerBound + i;
                    reachFirst = true;
                    initialCriticalPoint = lowerBound + i;
                    break;
                }
            }
        }

        saveInitialPopulationToFile(population, initialPopulationFilePath);

        return population;
    }

    private IndividualMultiObj createIndividual(List<Product> products) {
        IndividualMultiObj indiv = new IndividualMultiObj(products, lowerBound, upperBound);
        indiv.fitness(FITNESS_TYPE_ACCURATE);

        if (optimizationMode == MODE_AGE) {
            indiv.setUuid(UUID.randomUUID());
        }

        indiv.setOriginalObjectives(new double[]{
                indiv.getObjective(0),
                indiv.getObjective(1)
        });
        return indiv;
    }

    private boolean loadInitialPopulationFromFile(SolutionSet population, int expectedSize, String filePath) {
        File file = new File(filePath);
        if (!file.exists()) {
            return false;
        }

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
            @SuppressWarnings("unchecked")
            List<List<Product>> savedPopulation = (List<List<Product>>) ois.readObject();

            if (savedPopulation.size() != expectedSize) {
                System.out.println("Saved population size mismatch, creating new initial population");
                return false;
            }

            for (List<Product> products : savedPopulation) {
                population.add(createIndividual(products));
            }
            return true;
        } catch (Exception e) {
            System.err.println("Failed to load initial population: " + e.getMessage());
            return false;
        }
    }

    private void saveInitialPopulationToFile(SolutionSet population, String filePath) {
        try {
            List<List<Product>> productsList = new ArrayList<>();
            for (int i = 0; i < population.size(); i++) {
                IndividualMultiObj indiv = (IndividualMultiObj) population.get(i);
                productsList.add(new ArrayList<>(indiv.getProducts()));
            }

            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(filePath))) {
                oos.writeObject(productsList);
            }
        } catch (IOException e) {
            System.err.println("Failed to save initial population: " + e.getMessage());
        }
    }

    private double calculatePValue(SolutionSet union) {
        Set<String> uniqueConfigs = new HashSet<>();
        SolutionSet uniquePopulation = new SolutionSet(union.size());

        for (int i = 0; i < union.size(); i++) {
            IndividualMultiObj indiv = (IndividualMultiObj) union.get(i);
            String configKey = getConfigurationKey(indiv);
            if (!uniqueConfigs.contains(configKey)) {
                uniqueConfigs.add(configKey);
                uniquePopulation.add(indiv);
            }
        }

        NondominatedRanking ranking = new NondominatedRanking(uniquePopulation, false);
        int nonDominatedCount = ranking.getSubfront(0).size();

        return (double) nonDominatedCount / uniquePopulation.size();
    }

    private String getConfigurationKey(IndividualMultiObj indiv) {
        List<Integer> featureVector = indiv.getFeatureVector();
        return featureVector.toString();
    }

    private void updateBestSolution(SolutionSet population, int currentGeneration) {
        for (int i = 0; i < population.size(); i++) {
            IndividualMultiObj indiv = (IndividualMultiObj) population.get(i);
            double[] originalObjs = indiv.getOriginalObjectives();
            if (originalObjs[0] < bestOriginalObjective1) {
                bestOriginalObjective1 = originalObjs[0];
                bestOriginalObjective2 = originalObjs[1];
                bestGeneration = currentGeneration;
            }
        }
    }

    private void initializeCSVFile() throws IOException {
        File csvFile = new File(csvOutputPath);
        File parent = csvFile.getParentFile();
        if (parent != null && !parent.exists()) {
            if (!parent.mkdirs()) {
                throw new IOException("Failed to create directory: " + parent.getAbsolutePath());
            }
        }

        boolean fileExists = csvFile.exists();

        try (FileWriter writer = new FileWriter(csvOutputPath, true)) {
            if (!fileExists) {
                writer.write("NSGA-II Run for " + datasetName + " with mode " + modeStr + "\n");
                writer.write("Budget: " + evaluations + "\n\n");
            }
        }
    }

    private void writeGenerationInfoToCSV(int generation, double pValue, double bestOriginalObjective1) throws IOException {
        try (FileWriter writer = new FileWriter(csvOutputPath, true)) {
            writer.write(String.format("Generation %d p-value: %.4f best value: %.4f\n", generation, pValue,bestOriginalObjective1));
        }
    }

    private void writeFinalResultsToCSV() throws IOException {
        try (FileWriter writer = new FileWriter(csvOutputPath, true)) {
            writer.write("\n");
            writer.write(String.format("budget_used:%d\n", evaluations));
            writer.write(String.format("Running time: %.2f seconds\n\n",
                    (System.currentTimeMillis() - startTime) / 1000.0));

            writer.write(String.format(
                    "Best Solution: 'ft': %.6f, 'fa': %.6f appeared in Generation %d, p: %.4f\n",
                    -bestOriginalObjective1, bestOriginalObjective2, bestGeneration,
                    bestGeneration > 0 && bestGeneration <= pValuesHistory.size() ?
                            pValuesHistory.get(bestGeneration - 1) : 0.0
            ));

            if (bestGeneration > 0) {
                String pValuesStr = pValuesHistory.subList(0, bestGeneration).stream()
                        .map(p -> String.format("%.4f", p))
                        .collect(Collectors.joining(","));
                writer.write(String.format("p values until best solution: %s\n", pValuesStr));
            }
        }
        System.out.println("CSV will be saved to: " + new File(csvOutputPath).getAbsolutePath());
    }

    private void handlePopulationObjectives(SolutionSet solutionSet) {
        List<IndividualMultiObj> population = solutionSet.getSolutionsList().stream()
                .map(s -> (IndividualMultiObj) s)
                .collect(Collectors.toList());

        double[][] originalObjectives = new double[population.size()][2];
        for (int i = 0; i < population.size(); i++) {
            originalObjectives[i] = population.get(i).getOriginalObjectives();
        }
        double[][] normalizedOriginal = GenerateFA.normalizeOriginalObjectives(originalObjectives);

        if (optimizationMode == MODE_NOVELTY) {
            double[][] faValues = GenerateFA.generateNoveltyFA(
                    population,
                    normalizedOriginal,
                    originalObjectives,
                    noveltyArchive,
                    population.size()
            );
            setObjectivesFromFA(population, faValues);
            return;
        }

        maxGenerations = 12000 / population.size();

        if (optimizationMode == MODE_FT_FA) {
            applyOriginalObjectives(solutionSet);
            return;
        } else if (optimizationMode == MODE_G1_G2) {
            applyNormalizationAndCombineObjectives(solutionSet, normalizedOriginal);
            return;
        }

        String modeStr = getModeStringForGenerateFA(optimizationMode);
        double[][] faValues = GenerateFA.generate(
                population,
                modeStr,
                originalObjectives,
                (optimizationMode == MODE_AGE) ? ageInfo : null,
                (optimizationMode == MODE_NOVELTY) ?
                        noveltyArchive.stream()
                                .map(GenerateFA.ArchiveEntry::getIndividual)
                                .collect(Collectors.toList()) : null,
                currentGeneration,
                maxGenerations,
                random,
                currentFilePath
        );

        setObjectivesFromFA(population, faValues);
    }

    private void setObjectivesFromFA(List<IndividualMultiObj> population, double[][] faValues) {
        for (int i = 0; i < population.size(); i++) {
            IndividualMultiObj indiv = population.get(i);
            indiv.setObjective(0, faValues[i][0]);
            indiv.setObjective(1, faValues[i][1]);
        }
    }

    private String getModeStringForGenerateFA(int mode) {
        switch (mode) {
            case MODE_AGE: return GenerateFA.MODE_AGE;
            case MODE_GAUSSIAN: return GenerateFA.MODE_GAUSSIAN;
            case MODE_RECIPROCAL: return GenerateFA.MODE_RECIPROCAL;
            case MODE_PENALTY: return GenerateFA.MODE_PENALTY;
            case MODE_NOVELTY: return GenerateFA.MODE_NOVELTY;
            case MODE_DIVERSITY: return GenerateFA.MODE_DIVERSITY;
            default: throw new IllegalArgumentException("Unsupported mode: " + mode);
        }
    }

    private void applyOriginalObjectives(SolutionSet solutionSet) {
        List<IndividualMultiObj> population = solutionSet.getSolutionsList().stream()
                .map(s -> (IndividualMultiObj) s)
                .collect(Collectors.toList());

        for (IndividualMultiObj indiv : population) {
            double[] originalObjs = indiv.getOriginalObjectives();
            if (originalObjs == null) {
                originalObjs = new double[]{indiv.getObjective(0), indiv.getObjective(1)};
                indiv.setOriginalObjectives(originalObjs);
            }
            indiv.setObjective(0, originalObjs[0]);
            indiv.setObjective(1, originalObjs[1]);
        }
    }

    private void applyNormalizationAndCombineObjectives(SolutionSet solutionSet, double[][] normalizedOriginal) {
        List<IndividualMultiObj> population = solutionSet.getSolutionsList().stream()
                .map(s -> (IndividualMultiObj) s)
                .collect(Collectors.toList());

        for (int i = 0; i < population.size(); i++) {
            IndividualMultiObj indiv = population.get(i);
            double norm1 = normalizedOriginal[i][0];
            double norm2 = normalizedOriginal[i][1];

            double g1 = norm1 + norm2;
            double g2 = norm1 - norm2;

            indiv.setObjective(0, g1);
            indiv.setObjective(1, g2);
        }
    }

    private String getFullModeName(int mode) {
        switch (mode) {
            case MODE_FT_FA: return "ft_fa";
            case MODE_G1_G2: return "g1_g2";
            case MODE_AGE: return "age_maximization_fa";
            case MODE_GAUSSIAN: return "gaussian_fa";
            case MODE_RECIPROCAL: return "reciprocal_fa";
            case MODE_PENALTY: return "penalty_fa";
            case MODE_NOVELTY: return "novelty_maximization_fa";
            case MODE_DIVERSITY: return "diversity_fa";
            default: return "unknown_mode";
        }
    }

    private static String getModeString(int mode) {
        switch (mode) {
            case MODE_FT_FA: return "ft_fa";
            case MODE_G1_G2: return "g1_g2";
            case MODE_AGE: return "age";
            case MODE_GAUSSIAN: return "gaussian";
            case MODE_RECIPROCAL: return "reciprocal";
            case MODE_PENALTY: return "penalty";
            case MODE_NOVELTY: return "novelty";
            case MODE_DIVERSITY: return "diversity";
            default: return "unknown";
        }
    }

    void cleanupMemory() {
        try {
            System.out.println("Starting NSGA2 memory cleanup...");

            if (population != null) {
                population.clear();
                population = null;
            }
            if (ageInfo != null) {
                ageInfo.clear();
                ageInfo = null;
            }
            if (noveltyArchive != null) {
                noveltyArchive.clear();
                noveltyArchive = null;
            }
            if (pValuesHistory != null) {
                pValuesHistory.clear();
                pValuesHistory = null;
            }

            System.out.println("NSGA2 memory cleanup completed");
        } catch (Exception e) {
            System.err.println("Warning: Error during NSGA2 memory cleanup: " + e.getMessage());
        } finally {
            System.gc();
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            System.gc();
        }
    }

    public static void main(String[] args) throws Exception {
        // Default dataset and paths (kept as in original hard-coded values)
        java.nio.file.Path cwd = java.nio.file.Paths.get("").toAbsolutePath().normalize();
        System.out.println("Current working directory: " + cwd.toString());
        String[] datasets = {
                "Polly",
                "7z",
                "Amazon",
                "BerkeleyDBC",
                "CocheEcologico",
                "CounterStrikeSimpleFeatureModel",
                "Drupal",
                "DSSample",
                "Dune",
                "ElectronicDrum",
                "HiPAcc",
                "JavaGC",
                "JHipster",
                "lrzip",
                "ModelTransformation",
                "SmartHomev2.2",
                "SPLSSimuelESPnP",
                "VideoPlayer",
                "VP9",
                "WebPortal",
                "X264"
        };
        String basePath = workingDir+"/../../Datasets/";
        String outputDir = workingDir+"/../../Results/SPLT/";
        // Keep these hard-coded as requested
        int runs = 10;
        int nbProds = 11;
        String samplingMethod = "SAT4J";
        // Per your request: do NOT expose/use the seed-for-initial-solutions flag;
        // always hard-code to false.
        boolean useSeed = false;

        // Keep mode order and names consistent with the Python script's MODES
        int[] defaultOptimizationModes = {
                MODE_FT_FA,              // "ft_fa"
                MODE_PENALTY,            // "penalty_fa"
                MODE_G1_G2,              // "g1_g2"
                MODE_GAUSSIAN,           // "gaussian_fa"
                MODE_RECIPROCAL,         // "reciprocal_fa"
                MODE_AGE,                // "age_maximization_fa"
                MODE_NOVELTY,            // "novelty_maximization_fa"
                MODE_DIVERSITY           // "diversity_fa"
        };

        // Multi-process config defaults (CPU default consistent with Python's CPU_CORES=80)
        MultiProcessConfig mpConfig = new MultiProcessConfig();
        mpConfig.useProcessIsolation = true;
        mpConfig.maxCpuCores = 50;

        // CLI parameters (consistent with Python names):
        // --use-parallel          (enable parallel execution)  [default: enabled]
        // --no-parallel           (disable parallel execution)
        // --cpu-cores=<N>         (number of worker processes; default 80)
        // --mode=<mode|all|csv>   (single mode name, comma-separated list, or 'all'; default 'all')
        // --seeds=<spec>          (seeds specification: single '5', csv '0,1,2' or range '0-9'; default 0-9)
        boolean useParallel = true; // default like Python's parser.set_defaults(use_parallel=True)
        String modeArg = null;
        String seedsArg = null;

        if (args != null && args.length > 0) {
            for (String arg : args) {
                if (arg.equals("--use-parallel")) {
                    useParallel = true;
                } else if (arg.equals("--no-parallel")) {
                    useParallel = false;
                } else if (arg.startsWith("--cpu-cores=")) {
                    try {
                        mpConfig.maxCpuCores = Integer.parseInt(arg.substring("--cpu-cores=".length()));
                    } catch (NumberFormatException ignored) {}
                } else if (arg.startsWith("--mode=")) {
                    modeArg = arg.substring("--mode=".length()).trim();
                } else if (arg.startsWith("--seeds=")) {
                    seedsArg = arg.substring("--seeds=".length()).trim();
                }
                // NOTE: intentionally DO NOT parse a --use-seed flag (useSeed is hardcoded false)
            }
        }

        // Parse seeds argument into a list of seed integers. Default: 0..9
        List<Integer> seedsList = new ArrayList<>();
        if (seedsArg == null || seedsArg.isEmpty()) {
            for (int i = 0; i <= 9; i++) seedsList.add(i);
        } else {
            try {
                String s = seedsArg.trim();
                if (s.contains("-")) {
                    String[] parts = s.split("-", 2);
                    int start = Integer.parseInt(parts[0].trim());
                    int end = Integer.parseInt(parts[1].trim());
                    if (end < start) {
                        throw new IllegalArgumentException("Invalid seed range: end < start");
                    }
                    for (int i = start; i <= end; i++) seedsList.add(i);
                } else if (s.contains(",")) {
                    String[] items = s.split(",");
                    for (String item : items) {
                        if (!item.trim().isEmpty()) seedsList.add(Integer.parseInt(item.trim()));
                    }
                } else {
                    seedsList.add(Integer.parseInt(s));
                }
            } catch (Exception e) {
                // fallback to default 0..9 on parse error
                seedsList.clear();
                for (int i = 0; i <= 9; i++) seedsList.add(i);
                System.err.printf("Failed to parse --seeds='%s', falling back to default 0..9%n", seedsArg);
            }
        }

        // number of runs per dataset follows seeds list length to be consistent with Python behavior
        runs = seedsList.size();

        // Parse modeArg into optimizationModes array. Strict matching to Python MODES exact names:
        // Allowed exact names: "ft_fa", "penalty_fa", "g1_g2", "gaussian_fa", "reciprocal_fa",
        // "age_maximization_fa", "novelty_maximization_fa", "diversity_fa"
        int[] optimizationModes = defaultOptimizationModes;
        if (modeArg != null && !modeArg.isEmpty()) {
            String m = modeArg.trim();
            if (m.equalsIgnoreCase("all")) {
                optimizationModes = defaultOptimizationModes;
            } else {
                String[] modeItems = m.split(",");
                List<Integer> parsedModes = new ArrayList<>();
                for (String mi : modeItems) {
                    String modeName = mi.trim();
                    if (modeName.isEmpty()) continue;
                    Integer modeValue = null;
                    // Strict, exact matches only
                    switch (modeName) {
                        case "ft_fa":
                            modeValue = MODE_FT_FA; break;
                        case "penalty_fa":
                            modeValue = MODE_PENALTY; break;
                        case "g1_g2":
                            modeValue = MODE_G1_G2; break;
                        case "gaussian_fa":
                            modeValue = MODE_GAUSSIAN; break;
                        case "reciprocal_fa":
                            modeValue = MODE_RECIPROCAL; break;
                        case "age_maximization_fa":
                            modeValue = MODE_AGE; break;
                        case "novelty_maximization_fa":
                            modeValue = MODE_NOVELTY; break;
                        case "diversity_fa":
                            modeValue = MODE_DIVERSITY; break;
                        default:
                            System.err.printf("Warning: Unknown mode '%s' ignored (allowed exact names: ft_fa, penalty_fa, g1_g2, gaussian_fa, reciprocal_fa, age_maximization_fa, novelty_maximization_fa, diversity_fa)%n",
                                    modeName);
                    }
                    if (modeValue != null) parsedModes.add(modeValue);
                }
                if (!parsedModes.isEmpty()) {
                    optimizationModes = parsedModes.stream().mapToInt(Integer::intValue).toArray();
                } else {
                    optimizationModes = defaultOptimizationModes;
                }
            }
        }

        // Apply useParallel to mpConfig.enableMultiProcess
        mpConfig.enableMultiProcess = useParallel;

        // Print configuration summary
        System.out.printf("=== Starting Multi-process NSGA2 Optimization Tasks ===%n");
        System.out.printf("Number of datasets: %d, Number of runs per dataset: %d, Modes: %d%n",
                datasets.length, runs, optimizationModes.length);
        System.out.printf("Total tasks: %d, Budget per task: %d, Population size: %d%n",
                datasets.length * runs * optimizationModes.length, 12000, 8);
        System.out.printf("Multi-process: %s, Process isolation: %s, CPU cores: %d%n",
                mpConfig.enableMultiProcess ? "Enabled" : "Disabled",
                mpConfig.useProcessIsolation ? "Enabled" : "Disabled",
                mpConfig.maxCpuCores);
        System.out.printf("Time limit (per task): %d hours, Memory limit: %d MB per process%n",
                mpConfig.taskTimeoutHours, mpConfig.maxMemoryPerProcessMB);
        System.out.printf("Start time: %s%n", new Date());

        // Adjust cpus if more than available
        int availableCpus = Runtime.getRuntime().availableProcessors();
        if (mpConfig.maxCpuCores > availableCpus) {
            System.out.printf("Warning: Specified CPU cores (%d) exceed available cores (%d), automatically adjusted to %d%n",
                    mpConfig.maxCpuCores, availableCpus, availableCpus);
            mpConfig.maxCpuCores = availableCpus;
        }

        File outputDirFile = new File(outputDir);
        if (!outputDirFile.exists()) {
            outputDirFile.mkdirs();
        }

        // Generate tasks using the same helper (generateAllTasks) which uses run indices; we'll override seedValue using seedsList
        List<TaskInfo> allTasks = generateAllTasks(datasets, basePath, outputDir, runs, nbProds, samplingMethod, useSeed, optimizationModes);

        // If seeds were explicitly provided we want the seedValue to come from seedsList rather than run index.
        // generateAllTasks uses run indices for seedValue; we will override seedValue accordingly.
        if (seedsList != null && !seedsList.isEmpty()) {
            List<TaskInfo> overridden = new ArrayList<>();
            int idx = 0;
            for (TaskInfo t : allTasks) {
                // compute which seed to assign based on idx within runs block
                // tasks are ordered by dataset -> run -> optimizationMode
                int posWithinDataset = idx % (runs * optimizationModes.length);
                int runIndex = posWithinDataset / optimizationModes.length;
                // choose seedValue from seedsList by runIndex
                int seedValue = seedsList.get(Math.min(runIndex, seedsList.size() - 1));
                overridden.add(new TaskInfo(t.dataset, t.fmFile, runIndex, seedValue, t.optimizationMode, t.seed));
                idx++;
            }
            allTasks = overridden;
        }

        if (!mpConfig.enableMultiProcess) {
            executeTasksSequentially(allTasks, useSeed, nbProds, samplingMethod, outputDir);
        } else {
            executeTasksInParallelWithIsolation(allTasks, useSeed, nbProds, samplingMethod, outputDir, mpConfig);
        }

        System.out.printf("%n=== All NSGA2 Tasks Completed ===%n");
        System.out.printf("End time: %s%n", new Date());
    }

    public static void runWithDefaultMultiProcess(String[] datasets, String basePath, String outputDir,
                                                  int runs, int nbProds, String samplingMethod,
                                                  boolean useSeed, int[] optimizationModes) throws Exception {
        MultiProcessConfig defaultConfig = new MultiProcessConfig();
        defaultConfig.useProcessIsolation = true;
        runMultiProcessNSGA2(datasets, basePath, outputDir, runs, nbProds,
                samplingMethod, useSeed, optimizationModes, defaultConfig);
    }

    public static void runWithCustomCpuCores(String[] datasets, String basePath, String outputDir,
                                             int runs, int nbProds, String samplingMethod,
                                             boolean useSeed, int[] optimizationModes, int maxCpuCores) throws Exception {
        MultiProcessConfig config = new MultiProcessConfig(maxCpuCores, true, 25);
        config.useProcessIsolation = true;
        runMultiProcessNSGA2(datasets, basePath, outputDir, runs, nbProds,
                samplingMethod, useSeed, optimizationModes, config);
    }

    public static void runSingleProcess(String[] datasets, String basePath, String outputDir,
                                        int runs, int nbProds, String samplingMethod,
                                        boolean useSeed, int[] optimizationModes) throws Exception {
        MultiProcessConfig config = new MultiProcessConfig(1, false, 25);
        config.useProcessIsolation = false;
        runMultiProcessNSGA2(datasets, basePath, outputDir, runs, nbProds,
                samplingMethod, useSeed, optimizationModes, config);
    }

    public static void runMultiProcessNoIsolation(String[] datasets, String basePath, String outputDir,
                                                  int runs, int nbProds, String samplingMethod,
                                                  boolean useSeed, int[] optimizationModes, int maxCpuCores) throws Exception {
        MultiProcessConfig config = new MultiProcessConfig(maxCpuCores, true, 25);
        config.useProcessIsolation = false;
        runMultiProcessNSGA2(datasets, basePath, outputDir, runs, nbProds,
                samplingMethod, useSeed, optimizationModes, config);
    }

    public long getEvaluations() {
        return evaluations;
    }

    public int getLowerBound() {
        return lowerBound;
    }

    public int getUpperBound() {
        return upperBound;
    }

    public int getInitialCriticalPoint() {
        return initialCriticalPoint;
    }

    public int getOptimizationMode() {
        return optimizationMode;
    }

    public List<GenerateFA.ArchiveEntry> getNoveltyArchive() {
        return new ArrayList<>(noveltyArchive);
    }
}

class NSGA2ProcessExecutor {
    public static void main(String[] args) {
        if (args.length < 9) {
            System.err.println("Usage: NSGA2ProcessExecutor <dataset> <fmFile> <run> <seedValue> <optimizationMode> <useSeed> <nbProds> <samplingMethod> <outputDir>");
            System.exit(1);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("NSGA2ProcessExecutor shutdown hook triggered");
            System.out.flush();
            System.err.flush();
        }));

        NSGA2Optimizer optimizer = null;

        try {
            String dataset = args[0];
            String fmFile = args[1];
            int run = Integer.parseInt(args[2]);
            long seedValue = Long.parseLong(args[3]);
            int optimizationMode = Integer.parseInt(args[4]);
            boolean useSeed = Boolean.parseBoolean(args[5]);
            int nbProds = Integer.parseInt(args[6]);
            String samplingMethod = args[7];
            String outputDir = args[8];

            System.out.printf("NSGA2ProcessExecutor starting for %s_run%d_mode%d_seed%d%n",
                    dataset, run, optimizationMode, seedValue);
            System.out.flush();

            MAP_test.getInstance().initializeModelSolvers(fmFile, 2);

            List<Product> seed = Collections.emptyList();
            if (useSeed) {
                String fmFileName = new File(fmFile).getName();
                String seedPath = outputDir + "SAT4J\\" + fmFileName + "\\Samples\\" + nbProds + "prods\\Products.0";
                File seedFile = new File(seedPath);

                if (!seedFile.exists()) {
                    System.out.printf("Generating seeds for %s...%n", dataset);
                    MAP_test.getInstance().generateSeeds(fmFile, outputDir, 10, nbProds, samplingMethod);
                }
                seed = MAP_test.getInstance().loadSeedsFromFile(seedPath);
                if (seed == null || seed.isEmpty()) {
                    throw new RuntimeException("Seed loading failed or is empty: " + seedPath);
                }
                System.out.printf("Loaded seed size: %d%n", seed.size());
            }

            optimizer = new NSGA2Optimizer(1, 8, 12000, optimizationMode, fmFile, seedValue, useSeed);

            try {
                optimizer.runNSGA2(seed, seedValue);

                System.out.printf("NSGA2ProcessExecutor completed for %s_run%d_mode%d_seed%d%n",
                        dataset, run, optimizationMode, seedValue);

                double bestFT = -optimizer.bestOriginalObjective1;
                int budgetUsed = (int) optimizer.getEvaluations();
                int bestGeneration = optimizer.bestGeneration;

                System.out.printf("BestFt=%.6f%n", bestFT);
                System.out.printf("budget_used:%d%n", budgetUsed);
                System.out.printf("Best Generation: %d%n", bestGeneration);
                System.out.printf("Termination Reason: completed%n");

                System.out.flush();
                System.err.flush();

            } catch (Exception e) {
                System.err.printf("NSGA2ProcessExecutor execution failed: %s%n", e.getMessage());
                e.printStackTrace();

                double bestFT = optimizer != null ? -optimizer.bestOriginalObjective1 : 0.0;
                int budgetUsed = optimizer != null ? (int) optimizer.getEvaluations() : 0;

                System.out.printf("BestFt=%.6f%n", bestFT);
                System.out.printf("budget_used:%d%n", budgetUsed);
                System.out.printf("Termination Reason: exception_occurred%n");

                System.out.flush();
                System.err.flush();
                System.exit(1);
            } finally {
                if (optimizer != null) {
                    try {
                        optimizer.cleanupMemory();
                        System.out.println("NSGA2ProcessExecutor: Memory cleanup completed");
                    } catch (Exception cleanupEx) {
                        System.err.println("Failed to cleanup optimizer: " + cleanupEx.getMessage());
                    }
                }
            }

        } catch (Exception e) {
            System.err.printf("NSGA2ProcessExecutor failed: %s%n", e.getMessage());
            e.printStackTrace();

            System.out.printf("BestFt=0.000000%n");
            System.out.printf("budget_used:0%n");
            System.out.printf("Termination Reason: initialization_failed%n");

            System.out.flush();
            System.err.flush();

            if (optimizer != null) {
                try {
                    optimizer.cleanupMemory();
                } catch (Exception cleanupEx) {
                }
            }

            System.exit(1);
        }
    }
}