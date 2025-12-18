package mmo;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.SobolSequenceGenerator;
import org.apache.commons.math3.random.HaltonSequenceGenerator;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import spl.MAP_test;
import spl.fm.Product;
import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

import spl.techniques.QD.IndividualMultiObj;
import mmo.GenerateFA;
import mmo.GenerateFA.ArchiveEntry;

public class ProductSampler {
    public static final String SOBOL = "sobol";
    public static final String HALTON = "halton";
    public static final String STRATIFIED = "stratified";
    public static final String LATIN_HYPERCUBE = "latin_hypercube";
    public static final String MONTE_CARLO = "monte_carlo";
    public static final String COVERING_ARRAY = "covering_array";

    public static final String MODE_G1_G2 = "g1_g2";
    public static final String MODE_NOVELTY = "novelty";
    public static final String MODE_AGE = "age";
    public static final String MODE_DIVERSITY = "diversity";
    public static final String MODE_FT_FA = "ft_fa";
    public static final String MODE_GAUSSIAN = "gaussian";
    public static final String MODE_RECIPROCAL = "reciprocal";
    public static final String MODE_PENALTY = "penalty";

    public static final File workingDir = new File(System.getProperty("user.dir"));
    private static final String INITIAL_POPULATION_CSV_DIR = workingDir + "/../../Results/initial_populations/Samples_multi/";
    private static final String SAMPLES_CSV_DIR = workingDir + "/../../Results/Samples_multi/";

    private static class MultiProcessConfig {
        int maxCpuCores = 50;
        boolean enableMultiProcess = true;
        long taskTimeoutHours = 24;
        boolean useProcessIsolation = true;
        long maxMemoryPerProcessMB = 2048;

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
        final String dimacsFile;
        final long seed;
        final String mode;
        final String samplingMethod;
        final boolean firstSample;

        public TaskInfo(String dataset, String dimacsFile, long seed, String mode, String samplingMethod, boolean firstSample) {
            this.dataset = dataset;
            this.dimacsFile = dimacsFile;
            this.seed = seed;
            this.mode = mode;
            this.samplingMethod = samplingMethod;
            this.firstSample = firstSample;
        }

        @Override
        public String toString() {
            return String.format("%s_seed%d_mode%s_method%s_firstSample%s", dataset, seed, mode, samplingMethod, firstSample);
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

    private List<IndividualMultiObj> initialPopulation;
    private int sampleSize;
    private int populationSize;
    private Random random;
    private String datasetName;
    private String mode;
    private int minProductsPerIndividual = 1;
    private int maxProductsPerIndividual;
    private int maxFeaturesPerProduct;
    private long seed;
    private boolean firstSample;

    private Map<Product, Integer> productIdMap = new HashMap<>();
    private int nextProductId = 1;

    private Map<IndividualMultiObj, double[]> normalizedObjectivesMap = new HashMap<>();
    private Map<IndividualMultiObj, double[]> g1g2Map = new HashMap<>();

    private List<List<Integer>> batchIndices;
    private List<List<IndividualMultiObj>> batches;

    public ProductSampler(String datasetName, String mode, int populationSize, int sampleSize,
                          int maxProductsPerIndividual, long seed, boolean firstSample) {
        this.datasetName = datasetName;
        this.mode = mode;
        this.populationSize = populationSize;
        this.sampleSize = sampleSize;
        this.maxProductsPerIndividual = maxProductsPerIndividual;
        this.seed = seed;
        this.random = new Random(seed);
        this.batchIndices = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.firstSample = firstSample;

        new File(INITIAL_POPULATION_CSV_DIR).mkdirs();
        new File(SAMPLES_CSV_DIR).mkdirs();
    }

    public ProductSampler(String datasetName, String mode, int populationSize, int sampleSize,
                          int maxProductsPerIndividual, long seed) {
        this(datasetName, mode, populationSize, sampleSize, maxProductsPerIndividual, seed, false);
    }

    private static TaskResult executeSingleTaskWithProcessIsolation(TaskInfo taskInfo,
                                                                    int populationSize,
                                                                    int sampleSize,
                                                                    int maxProducts,
                                                                    MultiProcessConfig mpConfig) {
        long startTime = System.currentTimeMillis();

        ProcessBuilder processBuilder = new ProcessBuilder();

        List<String> command = new ArrayList<>();

        long processTimeoutMs = Math.max(0L, mpConfig.taskTimeoutHours * 60L * 60L * 1000L);
        final long GRACE_PERIOD_MS = 60L * 60L * 1000L;

        command.add("java");
        command.add("-Dsampler.process.timeout.ms=" + processTimeoutMs);
        command.add("-Dsampler.process.grace.ms=" + GRACE_PERIOD_MS);
        command.add("-Xmx" + mpConfig.maxMemoryPerProcessMB + "m");
        command.add("-XX:+UseG1GC");
        command.add("-XX:MaxGCPauseMillis=200");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add("mmo.ProductSamplerProcessExecutor");
        command.add(taskInfo.dataset);
        command.add(taskInfo.dimacsFile);
        command.add(String.valueOf(taskInfo.seed));
        command.add(taskInfo.mode);
        command.add(taskInfo.samplingMethod);
        command.add(String.valueOf(populationSize));
        command.add(String.valueOf(sampleSize));
        command.add(String.valueOf(maxProducts));
        command.add(String.valueOf(taskInfo.firstSample));

        processBuilder.command(command);
        processBuilder.redirectErrorStream(true);

        Process process = null;
        StringBuilder output = new StringBuilder();
        StringBuilder errorOutput = new StringBuilder();

        try {
            System.out.printf("[ProductSampler Process Isolation] Starting isolated process for:  %s%n", taskInfo);
            System.out.printf("[ProductSampler Process Isolation] Command: %s%n", String.join(" ", command));

            process = processBuilder.start();
            BufferedReader stdoutReader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            BufferedReader stderrReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line;
            long processEndTime = startTime + processTimeoutMs;
            boolean processCompleted = false;

            while (System.currentTimeMillis() < processEndTime && !processCompleted) {
                try {
                    int exitCode = process.exitValue();
                    processCompleted = true;
                    while ((line = stdoutReader.readLine()) != null) {
                        output.append(line).append("\n");
                        System.out.printf("[Sampler-Process-stdout-%s] %s%n", taskInfo, line);
                    }

                    while ((line = stderrReader.readLine()) != null) {
                        errorOutput.append(line).append("\n");
                        System.out.printf("[Sampler-Process-stderr-%s] %s%n", taskInfo, line);
                    }
                    break;
                } catch (IllegalThreadStateException e) {
                    try {
                        if (stdoutReader.ready()) {
                            line = stdoutReader.readLine();
                            if (line != null) {
                                output.append(line).append("\n");
                                System.out.printf("[Sampler-Process-stdout-%s] %s%n", taskInfo, line);
                            }
                        }

                        if (stderrReader.ready()) {
                            line = stderrReader.readLine();
                            if (line != null) {
                                errorOutput.append(line).append("\n");
                                System.out.printf("[Sampler-Process-stderr-%s] %s%n", taskInfo, line);
                            }
                        }

                        Thread.sleep(100);
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        break;
                    } catch (IOException ioex) {
                        System.err.printf("[ProductSampler Process Isolation] IO error reading process output: %s%n", ioex.getMessage());
                    }
                }
            }

            long executionTime = System.currentTimeMillis() - startTime;

            if (System.currentTimeMillis() >= processEndTime && !processCompleted) {
                System.out.printf("[ProductSampler Process Isolation] Process reached configured timeout for: %s, attempting graceful termination (grace=%d ms)%n",
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
                            while (stdoutReader.ready() && (line = stdoutReader.readLine()) != null) {
                                output.append(line).append("\n");
                                System.out.printf("[Sampler-Process-stdout-%s] %s%n", taskInfo, line);
                            }
                            while (stderrReader.ready() && (line = stderrReader.readLine()) != null) {
                                errorOutput.append(line).append("\n");
                                System.out.printf("[Sampler-Process-stderr-%s] %s%n", taskInfo, line);
                            }
                        } catch (IOException ioe) {
                        }
                    }

                    try {
                        while ((line = stdoutReader.readLine()) != null) {
                            output.append(line).append("\n");
                            System.out.printf("[Sampler-Process-stdout-%s] %s%n", taskInfo, line);
                        }
                        while ((line = stderrReader.readLine()) != null) {
                            errorOutput.append(line).append("\n");
                            System.out.printf("[Sampler-Process-stderr-%s] %s%n", taskInfo, line);
                        }
                    } catch (IOException ioe) {
                    }

                    long totalExecution = System.currentTimeMillis() - startTime;

                    if (! exitedGracefully) {
                        System.out.printf("[ProductSampler Process Isolation] Grace period expired for: %s; forcing termination now%n", taskInfo);
                        process.destroyForcibly();
                        try {
                            process.waitFor(5, TimeUnit.SECONDS);
                        } catch (InterruptedException ie) {
                            Thread.currentThread().interrupt();
                        }
                    } else {
                        System.out.printf("[ProductSampler Process Isolation] Child exited gracefully during grace period for: %s%n", taskInfo);
                    }

                    Map<String, Object> timeoutResult = parseProcessOutput(output.toString());
                    if (! timeoutResult.containsKey("terminationReason")) {
                        timeoutResult.put("terminationReason", "time_limit_reached");
                    }

                    return new TaskResult(taskInfo, true, totalExecution,
                            "Normal termination:  process-timeout reached (graceful wait)", timeoutResult);

                } catch (Exception e) {
                    long execTime = System.currentTimeMillis() - startTime;
                    String errorMsg = "Failed during graceful termination:  " + e.getMessage();
                    System.err.printf("[ProductSampler Process Isolation] Graceful termination failed for %s: %s%n", taskInfo, errorMsg);
                    e.printStackTrace();

                    if (process != null) {
                        try {
                            process.destroyForcibly();
                        } catch (Exception ex) {
                        }
                    }

                    Map<String, Object> fallbackResult = parseProcessOutput(output.toString());
                    fallbackResult.putIfAbsent("terminationReason", "time_limit_reached");

                    return new TaskResult(taskInfo, true, execTime, errorMsg, fallbackResult);
                }
            }

            if (processCompleted) {
                int exitCode = process.exitValue();

                if (exitCode == 0) {
                    System.out.printf("[ProductSampler Process Isolation] Task completed successfully:  %s, Time: %.2f seconds%n",
                            taskInfo, executionTime / 1000.0);

                    Map<String, Object> result = parseProcessOutput(output.toString());
                    return new TaskResult(taskInfo, true, executionTime, null, result);
                } else {
                    String outputStr = output.toString();
                    String errorStr = errorOutput.toString();

                    if (outputStr.contains("time_limit_reached") || outputStr.contains("24-hour") ||
                            outputStr.contains("Normal termination")) {
                        System.out.printf("[ProductSampler Process Isolation] Task normal termination (timeout or normal): %s, Time: %.2f seconds%n",
                                taskInfo, executionTime / 1000.0);

                        Map<String, Object> result = parseProcessOutput(outputStr);
                        if (! result.containsKey("terminationReason")) {
                            result.put("terminationReason", "time_limit_reached");
                        }
                        return new TaskResult(taskInfo, true, executionTime, "Normal timeout/termination", result);
                    }

                    String errorMsg = String.format("Process exited with code %d.Output: %s, Error: %s",
                            exitCode, outputStr, errorStr);
                    System.err.printf("[ProductSampler Process Isolation] Task failed:  %s, Error: %s%n", taskInfo, errorMsg);
                    return new TaskResult(taskInfo, false, executionTime, errorMsg, null);
                }
            }

        } catch (IOException e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Process creation/IO failed: " + e.getMessage();
            System.err.printf("[ProductSampler Process Isolation] Task failed: %s, Error: %s%n", taskInfo, errorMsg);
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            System.err.printf("[ProductSampler Process Isolation] Stack trace: %s%n", sw.toString());

            if (process != null) {
                process.destroyForcibly();
            }

            return new TaskResult(taskInfo, false, executionTime, errorMsg, null);
        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Process execution failed: " + e.getClass().getSimpleName() + " - " + e.getMessage();
            System.err.printf("[ProductSampler Process Isolation] Task failed: %s, Error:  %s%n", taskInfo, errorMsg);
            e.printStackTrace();
            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);
            System.err.printf("[ProductSampler Process Isolation] Stack trace: %s%n", sw.toString());

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
                if (line.contains("Sampling completed:  ")) {
                    result.put("samplingCompleted", true);
                } else if (line.contains("Number of generated samples: ")) {
                    try {
                        String[] parts = line.split(":");
                        if (parts.length > 1) {
                            String countStr = parts[1].trim();
                            result.put("sampleCount", Integer.parseInt(countStr));
                        }
                    } catch (NumberFormatException e) {
                        System.err.println("Failed to parse sample count from line: " + line);
                    }
                } else if (line.contains("Termination Reason: ")) {
                    String[] parts = line.split("Termination Reason:");
                    if (parts.length > 1) {
                        result.put("terminationReason", parts[1].trim());
                    }
                }
            }

            if (! result.containsKey("terminationReason")) {
                if (output.contains("time_limit_reached") || output.contains("24-hour") ||
                        output.contains("timeout")) {
                    result.put("terminationReason", "time_limit_reached");
                } else if (result.containsKey("samplingCompleted")) {
                    result.put("terminationReason", "completed");
                } else {
                    result.put("terminationReason", "unknown");
                }
            }

            if (! result.containsKey("samplingCompleted")) {
                result.put("samplingCompleted", false);
            }

        } catch (Exception e) {
            System.err.println("Error parsing process output: " + e.getMessage());
            result.clear();
            result.put("terminationReason", "parse_error");
            result.put("samplingCompleted", false);
        }
        return result;
    }

    private static TaskResult executeSingleTaskInThread(TaskInfo taskInfo,
                                                        int populationSize,
                                                        int sampleSize,
                                                        int maxProducts) {
        long startTime = System.currentTimeMillis();
        try {
            System.out.printf("[ProductSampler Thread] Starting task execution:  %s%n", taskInfo);

            synchronized (MAP_test.class) {
                MAP_test.getInstance().initializeModelSolvers(taskInfo.dimacsFile, 2);
            }
            ProductSampler sampler = new ProductSampler(taskInfo.dataset, taskInfo.mode,
                    populationSize, sampleSize, maxProducts, taskInfo.seed, taskInfo.firstSample);

            List<IndividualMultiObj> samples = sampler.sample(taskInfo.samplingMethod);

            long executionTime = System.currentTimeMillis() - startTime;
            System.out.printf("[ProductSampler Thread] Task completed:  %s, Time: %.2f seconds, Samples: %d%n",
                    taskInfo, executionTime / 1000.0, samples.size());

            Map<String, Object> result = new HashMap<>();
            result.put("samplingCompleted", true);
            result.put("sampleCount", samples.size());
            result.put("terminationReason", "completed");

            return new TaskResult(taskInfo, true, executionTime, null, result);

        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Task execution failed:  " + (e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName());
            System.err.printf("[ProductSampler Thread] Task failed: %s, Error: %s%n", taskInfo, errorMsg);

            String fullErrorMsg = String.format("%s: %s", e.getClass().getSimpleName(), errorMsg);
            if (e.getCause() != null) {
                fullErrorMsg += " - Cause: " + e.getCause().getMessage();
            }

            return new TaskResult(taskInfo, false, executionTime, fullErrorMsg, null);
        }
    }

    private static void executeTasksInParallelWithIsolation(List<TaskInfo> tasks,
                                                            int populationSize,
                                                            int sampleSize,
                                                            int maxProducts,
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
                        Thread thread = new Thread(r, "ProductSampler-Process-Worker-" + threadNumber.getAndIncrement());
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
                    return executeSingleTaskWithProcessIsolation(task, populationSize, sampleSize, maxProducts, mpConfig);
                } else {
                    return executeSingleTaskInThread(task, populationSize, sampleSize, maxProducts);
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

                        System.out.printf("[ProductSampler Main Process] Task completed: %s (Time: %.1f seconds, Termination: %s)%n",
                                result.taskInfo, result.executionTimeMs / 1000.0, terminationReason);
                    } else {
                        failed++;
                        System.err.printf("[ProductSampler Main Process] Task failed: %s, Error: %s%n",
                                result.taskInfo, result.errorMessage);
                    }

                    if ((completed + failed) % Math.max(1, tasks.size() / 10) == 0) {
                        long elapsedTime = System.currentTimeMillis() - overallStartTime;
                        double avgTimePerTask = elapsedTime / (double)(completed + failed);
                        long estimatedRemaining = (long)(avgTimePerTask * (tasks.size() - completed - failed));

                        System.out.printf("[ProductSampler Main Process] Progress: %d/%d completed, %d failed, Elapsed: %.1f minutes, Estimated: %.1f minutes%n",
                                completed, tasks.size(), failed,
                                elapsedTime / 60000.0, estimatedRemaining / 60000.0);
                    }

                } catch (ExecutionException e) {
                    failed++;
                    Throwable cause = e.getCause();
                    String errorMsg = cause != null ? cause.getMessage() : e.getMessage();
                    System.err.printf("[ProductSampler Main Process] Task execution exception: %s%n", errorMsg);
                } catch (CancellationException e) {
                    failed++;
                    System.err.printf("[ProductSampler Main Process] Task cancelled%n");
                }
            }
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                    System.out.println("[ProductSampler Main Process] Some tasks are still running, forcing shutdown...");
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        long totalTime = System.currentTimeMillis() - overallStartTime;
        System.out.printf("%nProductSampler Parallel execution completed:  %d/%d succeeded, %d failed, Total time: %.2f minutes%n",
                completed, tasks.size(), failed, totalTime / 60000.0);
    }

    private static void executeTasksSequentially(List<TaskInfo> tasks,
                                                 int populationSize,
                                                 int sampleSize,
                                                 int maxProducts) {
        System.out.printf("Starting sequential execution of %d tasks...%n", tasks.size());

        long overallStartTime = System.currentTimeMillis();
        int completed = 0;
        int failed = 0;

        for (TaskInfo task : tasks) {
            System.out.printf("%n--- Executing Task %d/%d:  %s ---%n",
                    completed + failed + 1, tasks.size(), task);

            TaskResult result = executeSingleTaskInThread(task, populationSize, sampleSize, maxProducts);

            if (result.success) {
                completed++;
                String terminationReason = result.result != null ?
                        (String)result.result.get("terminationReason") : "unknown";
                System.out.printf("Task completed:  %s, Termination: %s%n", task, terminationReason);
            } else {
                failed++;
                System.err.printf("Task failed: %s, Error:  %s%n", task, result.errorMessage);
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

    private static List<TaskInfo> generateAllTasks(String[] datasets, String basePath,
                                                   long[] seeds, String[] modes, String[] methods,
                                                   boolean firstSample) {
        List<TaskInfo> tasks = new ArrayList<>();

        for (String dataset : datasets) {
            String dimacsFile = basePath + dataset + ".dimacs";

            for (long seed : seeds) {
                String[] effectiveModes = firstSample ? new String[]{MODE_G1_G2} : modes;

                for (String mode : effectiveModes) {
                    for (String method : methods) {
                        tasks.add(new TaskInfo(dataset, dimacsFile, seed, mode, method, firstSample));
                    }
                }
            }
        }

        return tasks;
    }

    public static void runMultiProcessSampling(String[] datasets, String basePath,
                                               long[] seeds, String[] modes, String[] methods,
                                               int populationSize, int sampleSize, int maxProducts,
                                               MultiProcessConfig mpConfig, boolean firstSample) throws Exception {

        System.out.printf("=== Starting Multi-process ProductSampler Tasks ===%n");
        System.out.printf("First Sample Mode: %s%n", firstSample ? "ENABLED (will force g1_g2 mode)" : "DISABLED");
        System.out.printf("Number of datasets: %d, Seeds: %d, Modes: %d, Methods: %d%n",
                datasets.length, seeds.length, modes.length, methods.length);

        int actualModeCount = firstSample ? 1 : modes.length;
        System.out.printf("Total tasks:  %d%n",
                datasets.length * seeds.length * actualModeCount * methods.length);

        System.out.printf("Multi-process:  %s, Process isolation: %s, CPU cores: %d%n",
                mpConfig.enableMultiProcess ? "Enabled" : "Disabled",
                mpConfig.useProcessIsolation ? "Enabled" : "Disabled",
                mpConfig.maxCpuCores);
        System.out.printf("Start time: %s%n", new Date());

        List<TaskInfo> allTasks = generateAllTasks(datasets, basePath, seeds, modes, methods, firstSample);

        if (! mpConfig.enableMultiProcess) {
            executeTasksSequentially(allTasks, populationSize, sampleSize, maxProducts);
        } else {
            executeTasksInParallelWithIsolation(allTasks, populationSize, sampleSize, maxProducts, mpConfig);
        }

        System.out.printf("%n=== All ProductSampler Tasks Completed ===%n");
        System.out.printf("End time: %s%n", new Date());
    }

    public static void runMultiProcessSampling(String[] datasets, String basePath,
                                               long[] seeds, String[] modes, String[] methods,
                                               int populationSize, int sampleSize, int maxProducts,
                                               MultiProcessConfig mpConfig) throws Exception {
        runMultiProcessSampling(datasets, basePath, seeds, modes, methods,
                populationSize, sampleSize, maxProducts, mpConfig, false);
    }

    public List<IndividualMultiObj> sample(String samplingMethod) throws Exception {
        loadOrGenerateInitialPopulation();
        calculateMaxFeaturesPerProduct();

        List<IndividualMultiObj> samples;

        if (firstSample || MODE_G1_G2.equals(mode)) {
            System.out.println("\n===== Executing G1_G2 Mode Sampling =====");
            System.out.printf("First Sample:  %s, Sampling Method: %s%n", firstSample, samplingMethod);

            switch (samplingMethod.toLowerCase()) {
                case SOBOL:
                    samples = sobolSampling();
                    break;
                case HALTON:
                    samples = haltonSampling();
                    break;
                case STRATIFIED:
                    samples = stratifiedSampling();
                    break;
                case LATIN_HYPERCUBE:
                    samples = latinHypercubeSampling();
                    break;
                case MONTE_CARLO:
                    samples = monteCarloSampling();
                    break;
                case COVERING_ARRAY:
                    samples = coveringArraySampling();
                    break;
                default:
                    throw new IllegalArgumentException("Unknown sampling method:  " + samplingMethod);
            }

            samples = deduplicateAndComplete(samples, sampleSize);

            samples.sort(Comparator.comparingDouble(ind -> ind.getOriginalObjectives()[0]));

            processGlobalNormalizationForG1G2(samples);

            printObjectiveRanges(samples, samplingMethod);

            saveSamplesAsCsv(samples, samplingMethod, "figure1");

            saveG1G2AsCsv(samples, samplingMethod, "figure2");

            System.out.println("===== G1_G2 Mode Sampling Completed =====\n");

            if (firstSample) {
                System.out.println("[First Sample] g1_g2 mode data generated and saved for both figure1 and figure2");
                System.out.println("[First Sample] You can now run other modes which will load this data");
            }

            return samples;
        } else {
            System.out.printf("\n===== Executing %s Mode Sampling =====\n", mode.toUpperCase());
            System.out.printf("Loading g1_g2 base data for sampling method: %s%n", samplingMethod);

            samples = loadG1G2Samples(samplingMethod);

            samples = adjustSampleSize(samples);

            reconstructBatches(samples);

            validateBatchSizes();

            processBatchesForOtherModes(samples);

            List<IndividualMultiObj> orderedSamples = new ArrayList<>();
            for (List<IndividualMultiObj> batch :  batches) {
                orderedSamples.addAll(batch);
            }

            saveSamplesAsCsv(orderedSamples, samplingMethod, "figure1");
            saveG1G2AsCsv(orderedSamples, samplingMethod, "figure2");

            System.out.printf("===== %s Mode Sampling Completed =====\n\n", mode.toUpperCase());

            return orderedSamples;
        }
    }

    private void printObjectiveRanges(List<IndividualMultiObj> samples, String samplingMethod) {
        if (samples.isEmpty()) {
            System.out.println("Sampling result is empty, cannot compute objective ranges");
            return;
        }

        double minFt = Double.MAX_VALUE;
        double maxFt = Double.MIN_VALUE;
        double minFa = Double.MAX_VALUE;
        double maxFa = Double.MIN_VALUE;

        for (IndividualMultiObj sample :  samples) {
            double[] objectives = sample.getOriginalObjectives();
            if (objectives.length < 2) {
                continue;
            }

            double ft = objectives[0];
            double fa = objectives[1];

            if (ft < minFt) minFt = ft;
            if (ft > maxFt) maxFt = ft;

            if (fa < minFa) minFa = fa;
            if (fa > maxFa) maxFa = fa;
        }

        System.out.println("\n===== " + samplingMethod + " sampling objective ranges =====");
        System.out.printf("Primary objective (ft) range: [%.6f, %.6f], span: %.6f%n",
                minFt, maxFt, maxFt - minFt);
        System.out.printf("Auxiliary objective (fa) range: [%.6f, %.6f], span: %.6f%n",
                minFa, maxFa, maxFa - minFa);
        System.out.println("===========================================\n");
    }

    private void reconstructBatches(List<IndividualMultiObj> samples) {
        batches.clear();
        batchIndices.clear();

        int totalSamples = samples.size();
        int batchSize = getBatchSize();
        int numBatches = (totalSamples + batchSize - 1) / batchSize;

        List<Integer> sortedIndices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            sortedIndices.add(i);
        }

        batchIndices = getBatchIndices(sortedIndices, batchSize, numBatches);

        for (List<Integer> indices : batchIndices) {
            List<IndividualMultiObj> batch = new ArrayList<>();
            for (int idx : indices) {
                batch.add(samples.get(idx));
            }
            batches.add(batch);
        }
    }

    private List<List<Integer>> getBatchIndices(List<Integer> sortedIndices, int batchSize, int numBatches) {
        List<List<Integer>> batchIndices = new ArrayList<>();
        for (int i = 0; i < numBatches; i++) {
            batchIndices.add(new ArrayList<>());
        }

        double reverseProb = 0.8;

        for (int idx : sortedIndices) {
            boolean assigned = false;
            int attempts = 0;

            while (!assigned && attempts < 3) {
                int preferredBatch;
                if (random.nextDouble() < reverseProb) {
                    double ratio = 1.0 - ((double) idx / sortedIndices.size());
                    preferredBatch = (int) (ratio * numBatches);
                    preferredBatch = Math.min(preferredBatch, numBatches - 1);
                } else {
                    preferredBatch = random.nextInt(numBatches);
                }

                if (batchIndices.get(preferredBatch).size() < batchSize) {
                    batchIndices.get(preferredBatch).add(idx);
                    assigned = true;
                }
                attempts++;
            }

            if (!assigned) {
                for (int b = 0; b < numBatches; b++) {
                    if (batchIndices.get(b).size() < batchSize) {
                        batchIndices.get(b).add(idx);
                        assigned = true;
                        break;
                    }
                }
            }
        }

        return batchIndices;
    }

    private void validateBatchSizes() {
        int expectedSize = getBatchSize();
        boolean isValid = true;

        for (int i = 0; i < batches.size(); i++) {
            List<IndividualMultiObj> batch = batches.get(i);
            if (i < batches.size() - 1 && batch.size() != expectedSize) {
                System.err.printf("Batch size validation failed:  batch %d size is %d, expected %d%n",
                        i + 1, batch.size(), expectedSize);
                isValid = false;
            }
        }

        if (!isValid) {
            throw new IllegalStateException("Batch sizes do not match expectation, please check batch generation logic");
        }

        System.out.printf("Batch size validation passed: total %d batches, expected size %d%n",
                batches.size(), expectedSize);
    }

    private void processGlobalNormalizationForG1G2(List<IndividualMultiObj> allSamples) {
        MinMaxScaler globalFtScaler = new MinMaxScaler();
        MinMaxScaler globalFaScaler = new MinMaxScaler();

        double[] allFtValues = new double[allSamples.size()];
        double[] allFaValues = new double[allSamples.size()];
        for (int i = 0; i < allSamples.size(); i++) {
            double[] obj = allSamples.get(i).getOriginalObjectives();
            allFtValues[i] = obj[0];
            allFaValues[i] = obj[1];
        }

        globalFtScaler.fit(allFtValues);
        globalFaScaler.fit(allFaValues);

        for (IndividualMultiObj sample : allSamples) {
            double[] original = sample.getOriginalObjectives();
            double normFt = globalFtScaler.transform(original[0]);
            double normFa = globalFaScaler.transform(original[1]);
            normalizedObjectivesMap.put(sample, new double[]{normFt, normFa});

            double g1 = normFt + normFa;
            double g2 = normFt - normFa;
            g1g2Map.put(sample, new double[]{g1, g2});
        }
    }

    private void processBatchesForOtherModes(List<IndividualMultiObj> allSamples) {
        int t = 1;
        List<ArchiveEntry> noveltyArchive = new ArrayList<>();
        int totalBatches = batches.size();

        double[][] allOriginalObjectives = allSamples.stream()
                .map(IndividualMultiObj::getOriginalObjectives)
                .toArray(double[][]::new);

        double[] allFtValues = new double[allSamples.size()];
        double[] allFaValues = new double[allSamples.size()];
        for (int i = 0; i < allSamples.size(); i++) {
            allFtValues[i] = allOriginalObjectives[i][0];
            allFaValues[i] = allOriginalObjectives[i][1];
        }

        MinMaxScaler globalFtScaler = new MinMaxScaler();
        MinMaxScaler globalFaScaler = new MinMaxScaler();
        globalFtScaler.fit(allFtValues);
        globalFaScaler.fit(allFaValues);

        Map<IndividualMultiObj, double[]> globalNormMap = new HashMap<>();
        for (int i = 0; i < allSamples.size(); i++) {
            double normFt = globalFtScaler.transform(allFtValues[i]);
            double normFa = globalFaScaler.transform(allFaValues[i]);
            globalNormMap.put(allSamples.get(i), new double[]{normFt, normFa});
        }

        for (List<IndividualMultiObj> batch : batches) {
            double[][] batchOriginalObjectives = batch.stream()
                    .map(IndividualMultiObj::getOriginalObjectives)
                    .toArray(double[][]::new);

            double[][] batchNormalized = new double[batch.size()][];
            for (int i = 0; i < batch.size(); i++) {
                batchNormalized[i] = globalNormMap.get(batch.get(i));
            }

            switch (mode) {
                case MODE_NOVELTY:
                    handleNoveltyModeBatch(batch, globalNormMap, batchOriginalObjectives, noveltyArchive, t, totalBatches);
                    break;
                case MODE_AGE:
                    handleAgeModeBatch(batch, globalNormMap, t, totalBatches, batchOriginalObjectives);
                    break;
                case MODE_DIVERSITY:
                    handleDiversityModeBatch(batch, globalNormMap, t, totalBatches, batchOriginalObjectives);
                    break;
                case MODE_FT_FA:
                    handleFtFaModeBatch(batch, globalNormMap);
                    break;
                case MODE_GAUSSIAN:
                    handleGaussianModeBatch(batch, globalNormMap, batchOriginalObjectives, t, totalBatches);
                    break;
                case MODE_RECIPROCAL:
                    handleReciprocalModeBatch(batch, globalNormMap, batchOriginalObjectives, t, totalBatches);
                    break;
                case MODE_PENALTY:
                    handlePenaltyModeBatch(batch, globalNormMap, batchOriginalObjectives, t, totalBatches);
                    break;
                default:
                    throw new IllegalArgumentException("Unsupported mode: " + mode);
            }

            t++;
        }
    }

    private int getBatchSize() {
        return 20;
    }

    private void calculateMaxFeaturesPerProduct() {
        maxFeaturesPerProduct = 0;
        for (IndividualMultiObj ind : initialPopulation) {
            for (Product p : ind.getProducts()) {
                maxFeaturesPerProduct = Math.max(maxFeaturesPerProduct, p.size());
            }
        }
        if (maxFeaturesPerProduct == 0) maxFeaturesPerProduct = 1;
    }

    private void loadOrGenerateInitialPopulation() throws Exception {
        String csvFile = INITIAL_POPULATION_CSV_DIR + datasetName + "_initial_individuals_seed_" + seed + ".csv";
        File file = new File(csvFile);

        if (file.exists()) {
            initialPopulation = loadIndividualsFromCsv(csvFile);
            System.out.println("Loaded initial population from CSV: " + csvFile);
        } else {
            if (MODE_G1_G2.equals(mode) || firstSample) {
                initialPopulation = generateAndDeduplicateInitialPopulation();
                saveInitialPopulationToCsv(csvFile);
                System.out.println("Saved initial population to CSV: " + csvFile);
            } else {
                throw new FileNotFoundException("Initial population file not found, please run g1_g2 mode or firstSample=true first:  " + csvFile);
            }
        }

        if (initialPopulation.size() != populationSize) {
            throw new IllegalStateException("Initial population size mismatch: expected " + populationSize +
                    ", got " + initialPopulation.size());
        }

        countAndPrintDuplicateProducts();
    }

    private void countAndPrintDuplicateProducts() {
        Map<Product, Integer> productCountMap = new HashMap<>();
        for (IndividualMultiObj individual : initialPopulation) {
            for (Product product : individual.getProducts()) {
                productCountMap.put(product, productCountMap.getOrDefault(product, 0) + 1);
            }
        }

        Map<Product, Integer> duplicateProducts = new HashMap<>();
        for (Map.Entry<Product, Integer> entry : productCountMap.entrySet()) {
            if (entry.getValue() > 2) {
                duplicateProducts.put(entry.getKey(), entry.getValue());
            }
        }

        initializeProductIdMap(initialPopulation);

        for (Map.Entry<Product, Integer> entry : duplicateProducts.entrySet()) {
            Product product = entry.getKey();
            Integer productId = productIdMap.get(product);
        }
    }

    private List<IndividualMultiObj> generateAndDeduplicateInitialPopulation() throws Exception {
        List<IndividualMultiObj> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            population.add(generateSingleIndividual());
            if ((i + 1) % 1000 == 0) {
                System.out.println("Generated " + (i + 1) + "/" + populationSize + " initial individuals");
            }
        }
        return deduplicateAndComplete(population, populationSize);
    }

    private IndividualMultiObj generateSingleIndividual() throws Exception {
        int productCount = minProductsPerIndividual + random.nextInt(maxProductsPerIndividual - minProductsPerIndividual + 1);
        Set<Product> uniqueProducts = new HashSet<>();
        while (uniqueProducts.size() < productCount) {
            Product p = MAP_test.getInstance().getOneRandomProductSAT4J();
            uniqueProducts.add(p);
        }
        List<Product> products = new ArrayList<>(uniqueProducts);
        IndividualMultiObj individual = new IndividualMultiObj(products, minProductsPerIndividual, maxProductsPerIndividual);
        individual.fitness(IndividualMultiObj.FITNESS_TYPE_ACCURATE);
        individual.setOriginalObjectives(new double[]{
                individual.getObjective(0),
                individual.getObjective(1)
        });
        return individual;
    }

    private void initializeProductIdMap(List<IndividualMultiObj> individuals) {
        productIdMap.clear();
        nextProductId = 1;
        for (IndividualMultiObj ind : individuals) {
            for (Product p : ind.getProducts()) {
                if (! productIdMap.containsKey(p)) {
                    productIdMap.put(p, nextProductId++);
                }
            }
        }
    }

    private Set<Integer> getProductIdSet(IndividualMultiObj individual) {
        Set<Integer> idSet = new HashSet<>();
        for (Product p : individual.getProducts()) {
            Integer id = productIdMap.get(p);
            if (id != null) {
                idSet.add(id);
            } else {
                id = nextProductId++;
                productIdMap.put(p, id);
                idSet.add(id);
            }
        }
        return idSet;
    }

    private List<IndividualMultiObj> deduplicateIndividuals(List<IndividualMultiObj> individuals) {
        initializeProductIdMap(individuals);
        List<IndividualMultiObj> uniqueIndividuals = new ArrayList<>();
        List<Set<Integer>> uniqueIdSets = new ArrayList<>();

        for (IndividualMultiObj ind : individuals) {
            Set<Integer> currentIdSet = getProductIdSet(ind);
            boolean isDuplicate = false;

            for (Set<Integer> existingIdSet : uniqueIdSets) {
                Set<Integer> diff1 = new HashSet<>(currentIdSet);
                diff1.removeAll(existingIdSet);
                Set<Integer> diff2 = new HashSet<>(existingIdSet);
                diff2.removeAll(currentIdSet);
                if (diff1.isEmpty() && diff2.isEmpty()) {
                    isDuplicate = true;
                    break;
                }
            }

            if (! isDuplicate) {
                uniqueIndividuals.add(ind);
                uniqueIdSets.add(currentIdSet);
            }
        }
        return uniqueIndividuals;
    }

    private List<IndividualMultiObj> deduplicateAndComplete(List<IndividualMultiObj> individuals, int targetSize) throws Exception {
        List<IndividualMultiObj> current = new ArrayList<>(individuals);
        while (true) {
            List<IndividualMultiObj> unique = deduplicateIndividuals(current);
            if (unique.size() >= targetSize) {
                return unique.subList(0, targetSize);
            }
            int deficit = targetSize - unique.size();
            int toGenerate = deficit * 2;
            List<IndividualMultiObj> newIndividuals = new ArrayList<>();
            for (int i = 0; i < toGenerate; i++) {
                newIndividuals.add(generateSingleIndividual());
            }
            current = new ArrayList<>(unique);
            current.addAll(newIndividuals);
        }
    }

    private List<String> parseCsvLine(String line) {
        List<String> parts = new ArrayList<>();
        StringBuilder currentPart = new StringBuilder();
        boolean inQuotes = false;

        for (char c : line.toCharArray()) {
            if (c == '"') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                parts.add(currentPart.toString().trim());
                currentPart.setLength(0);
            } else {
                currentPart.append(c);
            }
        }
        parts.add(currentPart.toString().trim());
        return parts;
    }

    private List<IndividualMultiObj> loadG1G2Samples(String samplingMethod) throws Exception {
        String csvFile = getSampleCsvPath(samplingMethod, "figure1", MODE_G1_G2);
        System.out.println("Attempting to load g1_g2 mode sampling results:  " + csvFile);

        List<IndividualMultiObj> samples = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                List<String> parts = parseCsvLine(line);
                if (parts.size() < 4 + maxProductsPerIndividual * maxFeaturesPerProduct) continue;

                int targetStartIdx = parts.size() - 4;
                double originalFt = Double.parseDouble(parts.get(targetStartIdx));
                double originalFa = Double.parseDouble(parts.get(targetStartIdx + 1));

                List<Product> products = new ArrayList<>();
                int featureCount = 0;
                for (int p = 0; p < maxProductsPerIndividual; p++) {
                    Product product = new Product();
                    for (int f = 0; f < maxFeaturesPerProduct; f++) {
                        if (featureCount >= targetStartIdx) break;
                        String featureStr = parts.get(featureCount).trim();
                        if (! featureStr.isEmpty() && ! featureStr.equals("0")) {
                            product.add(Integer.parseInt(featureStr));
                        }
                        featureCount++;
                    }
                    if (! product.isEmpty()) {
                        products.add(product);
                    }
                }

                IndividualMultiObj individual = new IndividualMultiObj(products, minProductsPerIndividual, maxProductsPerIndividual);
                individual.setOriginalObjectives(new double[]{originalFt, originalFa});
                individual.fitness(IndividualMultiObj.FITNESS_TYPE_ACCURATE);
                samples.add(individual);
            }
        } catch (FileNotFoundException e) {
            throw new FileNotFoundException("g1_g2 sampling results not found, please run g1_g2 mode or firstSample=true to generate file: " + csvFile);
        }

        samples.sort(Comparator.comparingDouble(ind -> ind.getOriginalObjectives()[0]));
        return samples;
    }

    private List<IndividualMultiObj> loadIndividualsFromCsv(String csvFile) throws IOException {
        List<IndividualMultiObj> individuals = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line;
            boolean isFirstLine = true;
            while ((line = br.readLine()) != null) {
                if (isFirstLine) {
                    isFirstLine = false;
                    continue;
                }

                List<String> parts = parseCsvLine(line);
                if (parts.size() < 2 + minProductsPerIndividual) continue;

                double originalObj1 = Double.parseDouble(parts.get(0));
                double originalObj2 = Double.parseDouble(parts.get(1));

                List<Product> products = new ArrayList<>();
                for (int i = 2; i < parts.size(); i++) {
                    String productStr = parts.get(i).trim().replaceAll("^\"|\"$", "");
                    if (productStr.isEmpty()) continue;

                    Product p = new Product();
                    String[] featureStrs = productStr.split(",");
                    for (String fs : featureStrs) {
                        fs = fs.trim();
                        if (! fs.isEmpty() && !fs.equals("0")) {
                            p.add(Integer.parseInt(fs));
                        }
                    }
                    if (!p.isEmpty()) {
                        products.add(p);
                    }
                }

                IndividualMultiObj individual = new IndividualMultiObj(products, minProductsPerIndividual, maxProductsPerIndividual);
                individual.setOriginalObjectives(new double[]{originalObj1, originalObj2});
                individual.fitness(IndividualMultiObj.FITNESS_TYPE_ACCURATE);
                individuals.add(individual);
            }
        }
        return individuals;
    }

    private void saveInitialPopulationToCsv(String csvFile) throws IOException {
        try (PrintWriter writer = new PrintWriter(new File(csvFile))) {
            List<String> header = new ArrayList<>();
            header.add("OriginalObj1");
            header.add("OriginalObj2");
            for (int i = 1; i <= maxProductsPerIndividual; i++) {
                header.add("Product_" + i);
            }
            writer.println(String.join(",", header));

            for (IndividualMultiObj individual : initialPopulation) {
                String line = buildIndividualCsvLine(individual);
                writer.println(line);
            }
        }
    }

    private String buildIndividualCsvLine(IndividualMultiObj individual) {
        List<String> lineParts = new ArrayList<>();

        double[] originalObj = individual.getOriginalObjectives();
        lineParts.add(String.format("%.6f", originalObj[0]));
        lineParts.add(String.format("%.6f", originalObj[1]));

        List<Product> products = individual.getProducts();
        for (int i = 0; i < maxProductsPerIndividual; i++) {
            if (i < products.size()) {
                Product p = products.get(i);
                List<String> features = new ArrayList<>();
                for (int f :  p) {
                    features.add(String.valueOf(f));
                }
                while (features.size() < maxFeaturesPerProduct) {
                    features.add("0");
                }
                lineParts.add("\"" + String.join(",", features) + "\"");
            } else {
                List<String> zeros = new ArrayList<>();
                for (int j = 0; j < maxFeaturesPerProduct; j++) {
                    zeros.add("0");
                }
                lineParts.add("\"" + String.join(",", zeros) + "\"");
            }
        }

        return String.join(",", lineParts);
    }

    private void handleFtFaModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap) {
        for (IndividualMultiObj ind : batch) {
            double[] normalized = globalNormMap.get(ind);
            normalizedObjectivesMap.put(ind, normalized);

            double g1 = normalized[0] + normalized[1];
            double g2 = normalized[0] - normalized[1];
            g1g2Map.put(ind, new double[]{g1, g2});
        }
    }

    private void handleGaussianModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap,
                                         double[][] originalObjectives, int t, int tMax) {
        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_GAUSSIAN,
                originalObjectives,
                null,
                null,
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private void handleReciprocalModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap,
                                           double[][] originalObjectives, int t, int tMax) {
        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_RECIPROCAL,
                originalObjectives,
                null,
                null,
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private void handlePenaltyModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap,
                                        double[][] originalObjectives, int t, int tMax) {
        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_PENALTY,
                originalObjectives,
                null,
                null,
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private void handleNoveltyModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap,
                                        double[][] originalObjectives, List<ArchiveEntry> noveltyArchive, int t, int tMax) {
        int maxArchiveSize = 1000;

        GenerateFA.updateNoveltyArchive(batch, noveltyArchive, maxArchiveSize);

        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_NOVELTY,
                originalObjectives,
                null,
                noveltyArchive.stream().map(ArchiveEntry::getIndividual).collect(Collectors.toList()),
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private void handleAgeModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap, int t, int tMax, double[][] originalObjectives) {
        Map<UUID, Integer> ageInfo = new HashMap<>();
        int batchSize = getBatchSize();

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            if (ind.getUuid() == null) {
                ind.setUuid(UUID.randomUUID());
            }

            int age;
            if (t == 1) {
                age = i + 1;
            } else {
                age = batchSize + (t - 1);
            }

            ageInfo.put(ind.getUuid(), age);
        }

        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_AGE,
                originalObjectives,
                ageInfo,
                null,
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private void handleDiversityModeBatch(List<IndividualMultiObj> batch, Map<IndividualMultiObj, double[]> globalNormMap, int t, int tMax, double[][] originalObjectives) {
        double[][] faValues = GenerateFA.generate(
                batch,
                GenerateFA.MODE_DIVERSITY,
                originalObjectives,
                null,
                null,
                t,
                tMax,
                random,
                ""
        );

        for (int i = 0; i < batch.size(); i++) {
            IndividualMultiObj ind = batch.get(i);
            normalizedObjectivesMap.put(ind, globalNormMap.get(ind));
            g1g2Map.put(ind, new double[]{faValues[i][0], faValues[i][1]});
        }
    }

    private List<IndividualMultiObj> sobolSampling() {
        SobolSequenceGenerator sobol = new SobolSequenceGenerator(1);
        sobol.skipTo(populationSize);

        List<Double> sobolPoints = new ArrayList<>(sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            sobolPoints.add(sobol.nextVector()[0]);
        }

        return selectIndividualsByQuantiles(sobolPoints);
    }

    private List<IndividualMultiObj> haltonSampling() {
        int dimensions = 2;
        HaltonSequenceGenerator halton = new HaltonSequenceGenerator(dimensions);
        int skipCount = Math.max(populationSize, 1000);
        for (int i = 0; i < skipCount; i++) {
            halton.nextVector();
        }

        List<Double> haltonPoints = new ArrayList<>(sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            double point = halton.nextVector()[0];
            if (point < 0) point = 0.0;
            if (point >= 1) point = 0.999999999;
            haltonPoints.add(point);
        }

        return selectIndividualsByQuantiles(haltonPoints);
    }

    private List<IndividualMultiObj> stratifiedSampling() {
        int stratDims = Math.min(10, sampleSize);
        int strata = (int) Math.ceil(Math.pow(sampleSize, 1.0 / stratDims));
        int samplesPerStratum = (int) Math.ceil((double) sampleSize / (Math.pow(strata, stratDims)));

        List<IndividualMultiObj> samples = new ArrayList<>();
        for (int s = 0; s < strata; s++) {
            double start = (double) s / strata;
            double end = (double) (s + 1) / strata;
            for (int k = 0; k < samplesPerStratum; k++) {
                double point = start + random.nextDouble() * (end - start);
                int idx = (int) (point * populationSize);
                idx = Math.min(idx, populationSize - 1);
                samples.add(initialPopulation.get(idx));
                if (samples.size() >= sampleSize) {
                    return samples;
                }
            }
        }
        return samples;
    }

    public void forceGenerateInitialPopulation() throws Exception {
        String csvFile = INITIAL_POPULATION_CSV_DIR + datasetName +
                "_initial_individuals_seed_" + seed + ".csv";

        System.out.printf("  [Generating] %s (seed=%d) - Starting...%n", datasetName, seed);
        initialPopulation = generateAndDeduplicateInitialPopulation();
        calculateMaxFeaturesPerProduct();
        saveInitialPopulationToCsv(csvFile);

        System.out.printf("  [Success] %s (seed=%d) - Saved %d individuals to:  %s%n",
                datasetName, seed, initialPopulation.size(), csvFile);
    }

    private static void preGenerateAllInitialPopulations(String[] datasets, String basePath,
                                                         long[] seeds, int populationSize,
                                                         int maxProducts, MultiProcessConfig mpConfig)
            throws InterruptedException {
        int totalPopulations = datasets.length * seeds.length;

        System.out.printf("Total initial populations to process: %d%n", totalPopulations);
        System.out.printf("Population size per file: %d individuals%n", populationSize);
        System.out.printf("Max products per individual: %d%n", maxProducts);
        System.out.printf("Using process isolation with %d parallel workers%n%n", mpConfig.maxCpuCores);

        List<InitialPopulationTask> allTasks = new ArrayList<>();
        for (String dataset : datasets) {
            String dimacsFile = basePath + dataset + ".dimacs";
            for (long seed : seeds) {
                String csvFile = INITIAL_POPULATION_CSV_DIR + dataset +
                        "_initial_individuals_seed_" + seed + ".csv";
                allTasks.add(new InitialPopulationTask(dataset, dimacsFile, seed, csvFile));
            }
        }

        List<InitialPopulationTask> tasksToGenerate = new ArrayList<>();
        int alreadyExists = 0;

        for (InitialPopulationTask task : allTasks) {
            File file = new File(task.csvFile);
            if (file.exists() && file.length() > 0) {
                System.out.printf("[SKIP] %s (seed=%d) - File already exists%n",
                        task.dataset, task.seed);
                alreadyExists++;
            } else {
                tasksToGenerate.add(task);
            }
        }

        System.out.printf("%nInitial populations summary:%n");
        System.out.printf("  - Already exist: %d%n", alreadyExists);
        System.out.printf("  - Need to generate: %d%n", tasksToGenerate.size());

        if (tasksToGenerate.isEmpty()) {
            System.out.printf("%nAll initial populations already exist.Skipping Phase 0 generation.%n");
            return;
        }

        System.out.printf("%nStarting parallel generation of %d initial populations...%n%n",
                tasksToGenerate.size());

        executeInitialPopulationTasksWithProcessIsolation(tasksToGenerate, populationSize,
                maxProducts, mpConfig, alreadyExists,
                totalPopulations);
    }

    private static void executeInitialPopulationTasksWithProcessIsolation(
            List<InitialPopulationTask> tasks, int populationSize, int maxProducts,
            MultiProcessConfig mpConfig, int alreadyExists, int totalPopulations)
            throws InterruptedException {

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
                        Thread thread = new Thread(r, "InitPopGen-Process-Worker-" +
                                threadNumber.getAndIncrement());
                        thread.setDaemon(true);
                        return thread;
                    }
                },
                new ThreadPoolExecutor.CallerRunsPolicy()
        );

        CompletionService<InitialPopulationResult> completionService =
                new ExecutorCompletionService<>(executor);

        List<Future<InitialPopulationResult>> futures = new ArrayList<>();
        for (InitialPopulationTask task : tasks) {
            Future<InitialPopulationResult> future = completionService.submit(() ->
                    generateSingleInitialPopulationWithProcess(task, populationSize, maxProducts, mpConfig)
            );
            futures.add(future);
        }

        int generated = 0;
        int failed = 0;
        long startTime = System.currentTimeMillis();

        try {
            for (int i = 0; i < tasks.size(); i++) {
                try {
                    Future<InitialPopulationResult> future = completionService.poll(2, TimeUnit.HOURS);

                    if (future == null) {
                        System.err.printf("Timeout waiting for initial population generation task%n");
                        failed++;
                        continue;
                    }

                    InitialPopulationResult result = future.get();

                    if (result.success) {
                        generated++;
                        System.out.printf("[SUCCESS] %s (seed=%d) - Generated in %.1f seconds%n",
                                result.task.dataset, result.task.seed,
                                result.executionTimeMs / 1000.0);
                    } else {
                        failed++;
                        System.err.printf("[FAILED] %s (seed=%d) - Error: %s%n",
                                result.task.dataset, result.task.seed,
                                result.errorMessage);
                    }

                    int completed = generated + failed;
                    if (completed % 10 == 0 || completed == tasks.size()) {
                        long elapsed = System.currentTimeMillis() - startTime;
                        double avgTimePerTask = elapsed / (double) completed;
                        long estimatedRemaining = (long) (avgTimePerTask * (tasks.size() - completed));

                        System.out.printf("%n--- Progress: %d/%d (%.1f%%) - Generated: %d, Failed: %d, " +
                                        "Elapsed: %.1f min, Estimated:  %.1f min ---%n%n",
                                completed, tasks.size(),
                                (completed * 100.0 / tasks.size()),
                                generated, failed,
                                elapsed / 60000.0, estimatedRemaining / 60000.0);
                    }

                } catch (ExecutionException e) {
                    failed++;
                    Throwable cause = e.getCause();
                    String errorMsg = cause != null ? cause.getMessage() : e.getMessage();
                    System.err.printf("[EXCEPTION] Task execution failed: %s%n", errorMsg);
                } catch (CancellationException e) {
                    failed++;
                    System.err.printf("[CANCELLED] Task was cancelled%n");
                }
            }
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(1, TimeUnit.MINUTES)) {
                    System.out.println("Some tasks are still running, forcing shutdown...");
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }

        long totalTime = System.currentTimeMillis() - startTime;

        System.out.printf("%n========== Initial Population Generation Summary ==========%n");
        System.out.printf("  Total populations:        %d%n", totalPopulations);
        System.out.printf("  Already existed:         %d (%.1f%%)%n",
                alreadyExists, (alreadyExists * 100.0 / totalPopulations));
        System.out.printf("  Newly generated:         %d (%.1f%%)%n",
                generated, (generated * 100.0 / totalPopulations));
        System.out.printf("  Failed:                  %d (%.1f%%)%n",
                failed, (failed * 100.0 / totalPopulations));
        System.out.printf("  Total execution time:    %.2f minutes%n", totalTime / 60000.0);
        if (generated > 0) {
            System.out.printf("  Avg time per generation: %.1f seconds%n",
                    (totalTime / 1000.0) / generated);
        }
        System.out.printf("==========================================================%n%n");

        if (failed > 0) {
            System.err.printf("WARNING: %d initial populations failed to generate!%n", failed);
            System.err.printf("Please check the error messages above and retry if necessary.%n%n");
        }
    }

    private static InitialPopulationResult generateSingleInitialPopulationWithProcess(
            InitialPopulationTask task, int populationSize, int maxProducts,
            MultiProcessConfig mpConfig) {

        long startTime = System.currentTimeMillis();

        ProcessBuilder processBuilder = new ProcessBuilder();
        List<String> command = new ArrayList<>();

        command.add("java");
        command.add("-Xmx" + mpConfig.maxMemoryPerProcessMB + "m");
        command.add("-XX:+UseG1GC");
        command.add("-XX: MaxGCPauseMillis=200");
        command.add("-cp");
        command.add(System.getProperty("java.class.path"));
        command.add("mmo.InitialPopulationGenerator");
        command.add(task.dataset);
        command.add(task.dimacsFile);
        command.add(String.valueOf(task.seed));
        command.add(String.valueOf(populationSize));
        command.add(String.valueOf(maxProducts));
        command.add(task.csvFile);

        processBuilder.command(command);
        processBuilder.redirectErrorStream(true);

        Process process = null;
        StringBuilder output = new StringBuilder();

        try {
            System.out.printf("[START] %s (seed=%d) - Spawning process...%n",
                    task.dataset, task.seed);

            process = processBuilder.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                if (line.contains("SUCCESS") || line.contains("ERROR") || line.contains("Generated")) {
                    System.out.printf("  [%s-seed%d] %s%n", task.dataset, task.seed, line);
                }
            }

            int exitCode = process.waitFor();
            long executionTime = System.currentTimeMillis() - startTime;

            if (exitCode == 0) {
                File csvFile = new File(task.csvFile);
                if (csvFile.exists() && csvFile.length() > 0) {
                    return new InitialPopulationResult(task, true, executionTime, null);
                } else {
                    String errorMsg = "Process exited with 0 but file not found or empty";
                    return new InitialPopulationResult(task, false, executionTime, errorMsg);
                }
            } else {
                String errorMsg = "Process exited with code " + exitCode + ".Output: " +
                        output.toString().substring(0, Math.min(200, output.length()));
                return new InitialPopulationResult(task, false, executionTime, errorMsg);
            }

        } catch (IOException e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Process IO failed: " + e.getMessage();
            System.err.printf("[ERROR] %s (seed=%d) - %s%n", task.dataset, task.seed, errorMsg);

            if (process != null) {
                process.destroyForcibly();
            }

            return new InitialPopulationResult(task, false, executionTime, errorMsg);

        } catch (InterruptedException e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Process interrupted: " + e.getMessage();
            System.err.printf("[ERROR] %s (seed=%d) - %s%n", task.dataset, task.seed, errorMsg);

            if (process != null) {
                process.destroyForcibly();
            }

            Thread.currentThread().interrupt();
            return new InitialPopulationResult(task, false, executionTime, errorMsg);

        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = "Unexpected error: " + e.getClass().getSimpleName() + " - " + e.getMessage();
            System.err.printf("[ERROR] %s (seed=%d) - %s%n", task.dataset, task.seed, errorMsg);

            if (process != null) {
                process.destroyForcibly();
            }

            return new InitialPopulationResult(task, false, executionTime, errorMsg);

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
    }
    private static class InitialPopulationTask {
        final String dataset;
        final String dimacsFile;
        final long seed;
        final String csvFile;

        public InitialPopulationTask(String dataset, String dimacsFile, long seed, String csvFile) {
            this.dataset = dataset;
            this.dimacsFile = dimacsFile;
            this.seed = seed;
            this.csvFile = csvFile;
        }

        @Override
        public String toString() {
            return String.format("%s_seed%d", dataset, seed);
        }
    }

    private static class InitialPopulationResult {
        final InitialPopulationTask task;
        final boolean success;
        final long executionTimeMs;
        final String errorMessage;

        public InitialPopulationResult(InitialPopulationTask task, boolean success,
                                       long executionTimeMs, String errorMessage) {
            this.task = task;
            this.success = success;
            this.executionTimeMs = executionTimeMs;
            this.errorMessage = errorMessage;
        }
    }

    private static InitialPopulationResult generateSingleInitialPopulation(
            InitialPopulationTask task, int populationSize, int maxProducts) {

        long startTime = System.currentTimeMillis();

        try {
            System.out.printf("[START] %s (seed=%d) - Generating...%n", task.dataset, task.seed);

            synchronized (MAP_test.class) {
                MAP_test.getInstance().initializeModelSolvers(task.dimacsFile, 2);
            }

            ProductSampler sampler = new ProductSampler(
                    task.dataset, MODE_G1_G2, populationSize, 1000, maxProducts, task.seed, true
            );

            sampler.forceGenerateInitialPopulation();

            long executionTime = System.currentTimeMillis() - startTime;

            return new InitialPopulationResult(task, true, executionTime, null);

        } catch (Exception e) {
            long executionTime = System.currentTimeMillis() - startTime;
            String errorMsg = e.getMessage() != null ? e.getMessage() : e.getClass().getSimpleName();

            System.err.printf("[ERROR] %s (seed=%d) - Failed after %.1f seconds:  %s%n",
                    task.dataset, task.seed, executionTime / 1000.0, errorMsg);

            StringWriter sw = new StringWriter();
            PrintWriter pw = new PrintWriter(sw);
            e.printStackTrace(pw);

            return new InitialPopulationResult(task, false, executionTime, errorMsg);
        }
    }

    private List<IndividualMultiObj> latinHypercubeSampling() {
        List<Double> lhPoints = new ArrayList<>(sampleSize);
        double interval = 1.0 / sampleSize;

        JDKRandomGenerator apacheRandom = new JDKRandomGenerator();
        apacheRandom.setSeed(seed);
        UniformRealDistribution uniform = new UniformRealDistribution(apacheRandom, 0, interval);

        for (int i = 0; i < sampleSize; i++) {
            double point = i * interval + uniform.sample();
            lhPoints.add(point);
        }
        Collections.shuffle(lhPoints, random);

        return selectIndividualsByQuantiles(lhPoints);
    }

    private List<IndividualMultiObj> monteCarloSampling() {
        List<IndividualMultiObj> samples = new ArrayList<>(sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            samples.add(initialPopulation.get(random.nextInt(populationSize)));
        }
        return samples;
    }

    private List<IndividualMultiObj> coveringArraySampling() {
        List<Integer> allFeatures = MAP_test.getInstance().getAllFeatureIds();
        if (allFeatures.isEmpty()) {
            throw new IllegalStateException("No features available");
        }

        List<int[]> featurePairs = new ArrayList<>();
        for (int i = 0; i < allFeatures.size(); i++) {
            for (int j = i + 1; j < allFeatures.size(); j++) {
                featurePairs.add(new int[]{allFeatures.get(i), allFeatures.get(j)});
            }
        }

        Set<String> coveredPairs = new HashSet<>();
        List<IndividualMultiObj> samples = new ArrayList<>();

        while (samples.size() < sampleSize && ! featurePairs.isEmpty()) {
            int[] pair = featurePairs.get(random.nextInt(featurePairs.size()));
            boolean found = false;

            for (IndividualMultiObj individual : initialPopulation) {
                boolean coversFirst = individual.getProducts().stream()
                        .anyMatch(p -> p.contains(pair[0]) || p.contains(-pair[0]));
                boolean coversSecond = individual.getProducts().stream()
                        .anyMatch(p -> p.contains(pair[1]) || p.contains(-pair[1]));

                if (coversFirst && coversSecond) {
                    String pairKey = pair[0] + "_" + pair[1];
                    if (! coveredPairs.contains(pairKey)) {
                        samples.add(individual);
                        coveredPairs.add(pairKey);
                        featurePairs.remove(pair);
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                featurePairs.remove(pair);
            }
        }

        while (samples.size() < sampleSize) {
            samples.add(initialPopulation.get(random.nextInt(initialPopulation.size())));
        }

        return samples;
    }

    private List<IndividualMultiObj> selectIndividualsByQuantiles(List<Double> points) {
        Collections.sort(points);
        List<IndividualMultiObj> samples = new ArrayList<>(sampleSize);

        for (double p : points) {
            int idx = (int) (p * populationSize);
            idx = Math.min(idx, populationSize - 1);
            samples.add(initialPopulation.get(idx));
        }
        return samples;
    }

    private List<IndividualMultiObj> adjustSampleSize(List<IndividualMultiObj> samples) {
        if (MODE_G1_G2.equals(mode)) {
            return samples;
        }

        int batchSize = getBatchSize();
        int targetSize = (samples.size() / batchSize) * batchSize;
        if (targetSize < samples.size()) {
            return samples.subList(0, targetSize);
        }
        return samples;
    }

    private String getSampleCsvPath(String samplingMethod, String figureType) {
        return getSampleCsvPath(samplingMethod, figureType, this.mode);
    }

    private String getSampleCsvPath(String samplingMethod, String figureType, String mode) {
        return String.format("%ssampled_data_%s_%s_%s_%d_seed_%d_%s.csv",
                SAMPLES_CSV_DIR,
                datasetName,
                mode,
                samplingMethod,
                sampleSize,
                seed,
                figureType);
    }

    private void saveSamplesAsCsv(List<IndividualMultiObj> samples, String samplingMethod, String figureType) throws IOException {
        String csvFile = getSampleCsvPath(samplingMethod, figureType);

        try (PrintWriter writer = new PrintWriter(new File(csvFile))) {
            List<String> header = new ArrayList<>();
            for (int p = 1; p <= maxProductsPerIndividual; p++) {
                for (int f = 1; f <= maxFeaturesPerProduct; f++) {
                    header.add("Feature_" + p + "_" + f);
                }
            }
            header.add("original_ft");
            header.add("original_fa");
            header.add("normalized_ft");
            header.add("normalized_fa");
            writer.println(String.join(",", header));

            if (MODE_G1_G2.equals(mode) || firstSample) {
                for (IndividualMultiObj sample : samples) {
                    writeSampleToCsv(writer, sample);
                }
            } else {
                for (List<IndividualMultiObj> batch :  batches) {
                    for (IndividualMultiObj sample : batch) {
                        writeSampleToCsv(writer, sample);
                    }
                }
            }
        }
        System.out.println("Saved samples to CSV: " + csvFile);
    }

    private void writeSampleToCsv(PrintWriter writer, IndividualMultiObj sample) {
        List<String> lineParts = new ArrayList<>();

        List<Product> products = sample.getProducts();
        for (int p = 0; p < maxProductsPerIndividual; p++) {
            if (p < products.size()) {
                Product product = products.get(p);
                List<Integer> features = new ArrayList<>(product);
                while (features.size() < maxFeaturesPerProduct) {
                    features.add(0);
                }
                for (int f :  features) {
                    lineParts.add(String.valueOf(f));
                }
            } else {
                for (int f = 0; f < maxFeaturesPerProduct; f++) {
                    lineParts.add("0");
                }
            }
        }

        double[] original = sample.getOriginalObjectives();
        double[] normalized = normalizedObjectivesMap.get(sample);
        lineParts.add(String.format("%.6f", original[0]));
        lineParts.add(String.format("%.6f", original[1]));
        lineParts.add(String.format("%.6f", normalized[0]));
        lineParts.add(String.format("%.6f", normalized[1]));

        writer.println(String.join(",", lineParts));
    }

    private void saveG1G2AsCsv(List<IndividualMultiObj> samples, String samplingMethod, String figureType) throws IOException {
        String csvFile = getSampleCsvPath(samplingMethod, figureType);

        try (PrintWriter writer = new PrintWriter(new File(csvFile))) {
            List<String> header = new ArrayList<>();
            for (int p = 1; p <= maxProductsPerIndividual; p++) {
                for (int f = 1; f <= maxFeaturesPerProduct; f++) {
                    header.add("Feature_" + p + "_" + f);
                }
            }
            header.add("original_ft");
            header.add("original_fa");
            header.add("g1");
            header.add("g2");
            writer.println(String.join(",", header));

            if (MODE_G1_G2.equals(mode) || firstSample) {
                for (IndividualMultiObj sample : samples) {
                    writeG1G2ToCsv(writer, sample);
                }
            } else {
                for (List<IndividualMultiObj> batch : batches) {
                    for (IndividualMultiObj sample :  batch) {
                        writeG1G2ToCsv(writer, sample);
                    }
                }
            }
        }
        System.out.println("Saved g1/g2 to CSV: " + csvFile);
    }

    private void writeG1G2ToCsv(PrintWriter writer, IndividualMultiObj sample) {
        List<String> lineParts = new ArrayList<>();

        List<Product> products = sample.getProducts();
        for (int p = 0; p < maxProductsPerIndividual; p++) {
            if (p < products.size()) {
                Product product = products.get(p);
                List<Integer> features = new ArrayList<>(product);
                while (features.size() < maxFeaturesPerProduct) {
                    features.add(0);
                }
                for (int f : features) {
                    lineParts.add(String.valueOf(f));
                }
            } else {
                for (int f = 0; f < maxFeaturesPerProduct; f++) {
                    lineParts.add("0");
                }
            }
        }

        double[] original = sample.getOriginalObjectives();
        double[] g1g2 = g1g2Map.get(sample);
        lineParts.add(String.format("%.6f", original[0]));
        lineParts.add(String.format("%.6f", original[1]));
        lineParts.add(String.format("%.6f", g1g2[0]));
        lineParts.add(String.format("%.6f", g1g2[1]));

        writer.println(String.join(",", lineParts));
    }

    private static class MinMaxScaler {
        private double min;
        private double max;

        public void fit(double[] data) {
            if (data.length == 0) {
                min = 0;
                max = 0;
                return;
            }
            min = data[0];
            max = data[0];
            for (double d : data) {
                if (d < min) min = d;
                if (d > max) max = d;
            }
        }

        public double transform(double x) {
            if (max == min) return 0.5;
            return (x - min) / (max - min);
        }
    }

    public static void main(String[] args) {
        try {
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
            String basePath = workingDir + "/../../Datasets/";
            int populationSize = 10000;
            int sampleSize = 1000;
            int maxProducts = 8;

            String[] methods = {SOBOL, HALTON, STRATIFIED, LATIN_HYPERCUBE, MONTE_CARLO, COVERING_ARRAY};
            String[] defaultModes = {MODE_G1_G2, MODE_PENALTY, MODE_GAUSSIAN, MODE_RECIPROCAL,
                    MODE_AGE, MODE_NOVELTY, MODE_DIVERSITY};
            long[] defaultSeeds = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

            MultiProcessConfig mpConfig = new MultiProcessConfig();
            mpConfig.useProcessIsolation = true;
            mpConfig.maxCpuCores = 50;
            boolean useParallel = true;
            String modeArg = null;
            String seedsArg = null;
            boolean skipFirstSample = false;
            boolean skipPhase0 = false;

            if (args != null && args.length > 0) {
                for (String arg : args) {
                    if (arg.equals("--use-parallel")) {
                        useParallel = true;
                    } else if (arg.equals("--no-parallel")) {
                        useParallel = false;
                    } else if (arg.equals("--skip-first-sample")) {
                        skipFirstSample = true;
                    } else if (arg.equals("--skip-phase0")) {
                        skipPhase0 = true;
                    } else if (arg.startsWith("--cpu-cores=")) {
                        try {
                            mpConfig.maxCpuCores = Integer.parseInt(arg.substring("--cpu-cores=".length()));
                        } catch (NumberFormatException ignored) {}
                    } else if (arg.startsWith("--mode=")) {
                        modeArg = arg.substring("--mode=".length()).trim();
                    } else if (arg.startsWith("--seeds=")) {
                        seedsArg = arg.substring("--seeds=".length()).trim();
                    }
                }
            }

            List<Long> seedsList = new ArrayList<>();
            if (seedsArg == null || seedsArg.isEmpty()) {
                for (long s : defaultSeeds) seedsList.add(s);
            } else {
                try {
                    String s = seedsArg.trim();
                    if (s.contains("-")) {
                        String[] parts = s.split("-", 2);
                        long start = Long.parseLong(parts[0].trim());
                        long end = Long.parseLong(parts[1].trim());
                        if (end < start) {
                            throw new IllegalArgumentException("Invalid seed range: end < start");
                        }
                        for (long i = start; i <= end; i++) seedsList.add(i);
                    } else if (s.contains(",")) {
                        String[] items = s.split(",");
                        for (String item :  items) {
                            if (! item.trim().isEmpty()) seedsList.add(Long.parseLong(item.trim()));
                        }
                    } else {
                        seedsList.add(Long.parseLong(s));
                    }
                } catch (Exception e) {
                    seedsList.clear();
                    for (long sd : defaultSeeds) seedsList.add(sd);
                    System.err.printf("Failed to parse --seeds='%s', falling back to default 0..9%n", seedsArg);
                }
            }

            long[] seeds = seedsList.stream().mapToLong(Long::longValue).toArray();

            String[] modes = defaultModes;
            if (modeArg != null && !modeArg.isEmpty()) {
                String m = modeArg.trim();
                if (m.equalsIgnoreCase("all")) {
                    modes = defaultModes;
                } else {
                    String[] modeItems = m.split(",");
                    List<String> parsedModes = new ArrayList<>();
                    for (String mi : modeItems) {
                        String modeName = mi.trim();
                        if (modeName.isEmpty()) continue;
                        switch (modeName) {
                            case "g1_g2":
                            case "penalty":
                            case "gaussian":
                            case "reciprocal":
                            case "age":
                            case "novelty":
                            case "diversity":
                            case "ft_fa":
                                parsedModes.add(modeName);
                                break;
                            default:
                                System.err.printf("Warning: Unknown mode '%s' ignored%n", modeName);
                        }
                    }
                    if (! parsedModes.isEmpty()) {
                        modes = parsedModes.toArray(new String[0]);
                    } else {
                        modes = defaultModes;
                    }
                }
            }

            mpConfig.enableMultiProcess = useParallel;

            int availableCpus = Runtime.getRuntime().availableProcessors();
            if (mpConfig.maxCpuCores > availableCpus) {
                System.out.printf("Warning: Specified CPU cores (%d) exceed available cores (%d), automatically adjusted to %d%n",
                        mpConfig.maxCpuCores, availableCpus, availableCpus);
                mpConfig.maxCpuCores = availableCpus;
            }

            System.out.printf("===========================================================%n");
            System.out.printf("=== ProductSampler Three-Phase Execution ===%n");
            System.out.printf("===========================================================%n");
            System.out.printf("Configuration:%n");
            System.out.printf("  - Datasets: %d%n", datasets.length);
            System.out.printf("  - Seeds: %d%n", seeds.length);
            System.out.printf("  - Modes: %d (%s)%n", modes.length, String.join(", ", modes));
            System.out.printf("  - Sampling methods: %d (%s)%n", methods.length, String.join(", ", methods));
            System.out.printf("  - Population size: %d%n", populationSize);
            System.out.printf("  - Sample size: %d%n", sampleSize);
            System.out.printf("  - Max products: %d%n", maxProducts);
            System.out.printf("  - Multi-process: %s%n", mpConfig.enableMultiProcess ? "Enabled" : "Disabled");
            System.out.printf("  - Process isolation: %s%n", mpConfig.useProcessIsolation ?  "Enabled" : "Disabled");
            System.out.printf("  - CPU cores: %d%n", mpConfig.maxCpuCores);
            System.out.printf("Overall start time: %s%n", new Date());
            System.out.printf("===========================================================%n%n");

            if (! skipPhase0) {
                System.out.printf("===========================================================%n");
                System.out.printf("=== PHASE 0: Generate Initial Populations ===%n");
                System.out.printf("===========================================================%n");
                System.out.printf("This phase will pre-generate all initial populations%n");
                System.out.printf("to avoid concurrency issues in later phases.%n");
                System.out.printf("Initial populations needed: %d (datasets) × %d (seeds) = %d%n",
                        datasets.length, seeds.length, datasets.length * seeds.length);
                System.out.printf("Phase 0 start time: %s%n%n", new Date());

                long phase0Start = System.currentTimeMillis();
                preGenerateAllInitialPopulations(datasets, basePath, seeds, populationSize, maxProducts, mpConfig);
                long phase0Duration = System.currentTimeMillis() - phase0Start;

                System.out.printf("===========================================================%n");
                System.out.printf("=== PHASE 0 COMPLETED ===%n");
                System.out.printf("===========================================================%n");
                System.out.printf("Phase 0 end time: %s%n", new Date());
                System.out.printf("Phase 0 duration: %.2f minutes%n%n", phase0Duration / 60000.0);

                try {
                    System.out.println("Waiting 5 seconds before starting Phase 1...\n");
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            } else {
                System.out.println("[SKIP] Phase 0 (Initial Population Generation) skipped due to --skip-phase0 flag\n");
            }

            if (!skipFirstSample) {
                System.out.printf("===========================================================%n");
                System.out.printf("=== PHASE 1: First Sample (g1_g2) ===%n");
                System.out.printf("===========================================================%n");
                System.out.printf("This phase will generate base g1_g2 data for all datasets and seeds%n");
                System.out.printf("Tasks in Phase 1: %d (datasets) × %d (seeds) × %d (methods) = %d%n",
                        datasets.length, seeds.length, methods.length,
                        datasets.length * seeds.length * methods.length);
                System.out.printf("Phase 1 start time: %s%n%n", new Date());

                long phase1Start = System.currentTimeMillis();
                runMultiProcessSampling(datasets, basePath, seeds,
                        new String[]{MODE_G1_G2},
                        methods, populationSize, sampleSize, maxProducts, mpConfig, true);
                long phase1Duration = System.currentTimeMillis() - phase1Start;

                System.out.printf("%n===========================================================%n");
                System.out.printf("=== PHASE 1 COMPLETED ===%n");
                System.out.printf("===========================================================%n");
                System.out.printf("Phase 1 end time: %s%n", new Date());
                System.out.printf("Phase 1 duration: %.2f minutes%n%n", phase1Duration / 60000.0);

                try {
                    System.out.println("Waiting 5 seconds before starting Phase 2...%n");
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            } else {
                System.out.println("[SKIP] Phase 1 (First Sample) skipped due to --skip-first-sample flag%n");
            }

            System.out.printf("===========================================================%n");
            System.out.printf("=== PHASE 2: Other Modes ===%n");
            System.out.printf("===========================================================%n");
            System.out.printf("This phase will process all specified modes using g1_g2 base data%n");

            List<String> otherModes = new ArrayList<>();
            for (String mode : modes) {
                if (!MODE_G1_G2.equals(mode)) {
                    otherModes.add(mode);
                }
            }

            if (otherModes.isEmpty()) {
                System.out.println("No other modes to execute.Only g1_g2 was specified or executed.%n");
            } else {
                String[] otherModesArray = otherModes.toArray(new String[0]);
                System.out.printf("Modes to execute: %s%n", String.join(", ", otherModesArray));
                System.out.printf("Tasks in Phase 2: %d (datasets) × %d (seeds) × %d (modes) × %d (methods) = %d%n",
                        datasets.length, seeds.length, otherModesArray.length, methods.length,
                        datasets.length * seeds.length * otherModesArray.length * methods.length);
                System.out.printf("Phase 2 start time:  %s%n%n", new Date());

                long phase2Start = System.currentTimeMillis();
                runMultiProcessSampling(datasets, basePath, seeds, otherModesArray,
                        methods, populationSize, sampleSize, maxProducts, mpConfig, false);
                long phase2Duration = System.currentTimeMillis() - phase2Start;

                System.out.printf("%n===========================================================%n");
                System.out.printf("=== PHASE 2 COMPLETED ===%n");
                System.out.printf("===========================================================%n");
                System.out.printf("Phase 2 end time: %s%n", new Date());
                System.out.printf("Phase 2 duration: %.2f minutes%n%n", phase2Duration / 60000.0);
            }

            System.out.printf("===========================================================%n");
            System.out.printf("=== ALL PHASES COMPLETED ===%n");
            System.out.printf("===========================================================%n");
            System.out.printf("Overall end time: %s%n", new Date());
            System.out.printf("Phase 0 (Initial Populations): %s%n", skipPhase0 ? "SKIPPED" : "COMPLETED");
            System.out.printf("Phase 1 (First Sample): %s%n", skipFirstSample ? "SKIPPED" : "COMPLETED");
            System.out.printf("Phase 2 (Other Modes): %s%n", otherModes.isEmpty() ? "NO TASKS" : "COMPLETED");
            System.out.printf("Total modes executed: %s%n",
                    skipFirstSample ?  String.format("%d modes", otherModes.size()) :
                            String.format("g1_g2 + %d other modes", otherModes.size()));
            System.out.printf("===========================================================%n");

        } catch (Exception e) {
            System.err.println("Fatal error in main execution:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    public static void runWithDefaultMultiProcess(String[] datasets, String basePath,
                                                  long[] seeds, String[] modes, String[] methods,
                                                  int populationSize, int sampleSize, int maxProducts) throws Exception {
        MultiProcessConfig defaultConfig = new MultiProcessConfig();
        defaultConfig.useProcessIsolation = true;
        runMultiProcessSampling(datasets, basePath, seeds, modes, methods,
                populationSize, sampleSize, maxProducts, defaultConfig, false);
    }

    public static void runWithCustomCpuCores(String[] datasets, String basePath,
                                             long[] seeds, String[] modes, String[] methods,
                                             int populationSize, int sampleSize, int maxProducts,
                                             int maxCpuCores) throws Exception {
        MultiProcessConfig config = new MultiProcessConfig(maxCpuCores, true, 24);
        config.useProcessIsolation = true;
        runMultiProcessSampling(datasets, basePath, seeds, modes, methods,
                populationSize, sampleSize, maxProducts, config, false);
    }

    public static void runSingleProcess(String[] datasets, String basePath,
                                        long[] seeds, String[] modes, String[] methods,
                                        int populationSize, int sampleSize, int maxProducts) throws Exception {
        MultiProcessConfig config = new MultiProcessConfig(1, false, 24);
        config.useProcessIsolation = false;
        runMultiProcessSampling(datasets, basePath, seeds, modes, methods,
                populationSize, sampleSize, maxProducts, config, false);
    }
}

class ProductSamplerProcessExecutor {
    public static void main(String[] args) {
        if (args.length < 9) {
            System.err.println("Usage: ProductSamplerProcessExecutor <dataset> <dimacsFile> <seed> <mode> <samplingMethod> <populationSize> <sampleSize> <maxProducts> <firstSample>");
            System.exit(1);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("ProductSamplerProcessExecutor shutdown hook triggered");
            System.out.flush();
            System.err.flush();
        }));

        try {
            String dataset = args[0];
            String dimacsFile = args[1];
            long seed = Long.parseLong(args[2]);
            String mode = args[3];
            String samplingMethod = args[4];
            int populationSize = Integer.parseInt(args[5]);
            int sampleSize = Integer.parseInt(args[6]);
            int maxProducts = Integer.parseInt(args[7]);
            boolean firstSample = Boolean.parseBoolean(args[8]);

            System.out.printf("ProductSamplerProcessExecutor starting for %s_seed%d_mode%s_method%s_firstSample%s%n",
                    dataset, seed, mode, samplingMethod, firstSample);
            System.out.flush();

            MAP_test.getInstance().initializeModelSolvers(dimacsFile, 2);

            ProductSampler sampler = new ProductSampler(dataset, mode, populationSize, sampleSize, maxProducts, seed, firstSample);
            List<IndividualMultiObj> samples = sampler.sample(samplingMethod);

            System.out.printf("ProductSamplerProcessExecutor completed for %s_seed%d_mode%s_method%s_firstSample%s%n",
                    dataset, seed, mode, samplingMethod, firstSample);
            System.out.printf("Number of generated samples: %d%n", samples.size());
            System.out.printf("Sampling completed:  true%n");
            System.out.printf("Termination Reason: completed%n");
            System.out.flush();
            System.err.flush();
        } catch (Exception e) {
            System.err.printf("ProductSamplerProcessExecutor execution failed: %s%n", e.getMessage());
            e.printStackTrace();
            System.out.printf("Sampling completed: false%n");
            System.out.printf("Termination Reason: exception_occurred%n");
            System.out.flush();
            System.err.flush();
            System.exit(1);
        }
    }
}

class InitialPopulationGenerator {
    public static void main(String[] args) {
        if (args.length < 6) {
            System.err.println("Usage: InitialPopulationGenerator <dataset> <dimacsFile> <seed> <populationSize> <maxProducts> <csvFile>");
            System.exit(1);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("InitialPopulationGenerator shutdown hook triggered");
            System.out.flush();
            System.err.flush();
        }));

        try {
            String dataset = args[0];
            String dimacsFile = args[1];
            long seed = Long.parseLong(args[2]);
            int populationSize = Integer.parseInt(args[3]);
            int maxProducts = Integer.parseInt(args[4]);
            String csvFile = args[5];

            System.out.printf("InitialPopulationGenerator starting for %s (seed=%d)%n", dataset, seed);
            System.out.flush();

            MAP_test.getInstance().initializeModelSolvers(dimacsFile, 2);
            System.out.printf("Model solvers initialized for %s%n", dataset);

            ProductSampler sampler = new ProductSampler(
                    dataset, ProductSampler.MODE_G1_G2, populationSize, 1000, maxProducts, seed, true
            );

            sampler.forceGenerateInitialPopulation();

            System.out.printf("SUCCESS: Generated initial population for %s (seed=%d)%n", dataset, seed);
            System.out.printf("File saved to: %s%n", csvFile);
            System.out.flush();
            System.err.flush();

        } catch (Exception e) {
            System.err.printf("ERROR: Failed to generate initial population:  %s%n", e.getMessage());
            e.printStackTrace();
            System.out.flush();
            System.err.flush();
            System.exit(1);
        }
    }
}