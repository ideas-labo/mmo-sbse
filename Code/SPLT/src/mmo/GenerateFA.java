package mmo;

import spl.fm.Product;
import spl.techniques.QD.IndividualMultiObj;

import java.util.*;
import java.util.stream.Collectors;

public class GenerateFA {
    public static final String MODE_AGE = "age";
    public static final String MODE_GAUSSIAN = "gaussian";
    public static final String MODE_RECIPROCAL = "reciprocal";
    public static final String MODE_PENALTY = "penalty";
    public static final String MODE_NOVELTY = "novelty";
    public static final String MODE_DIVERSITY = "diversity";


    public static class ArchiveEntry {
        private final IndividualMultiObj individual;
        private final double noveltyScore;

        public ArchiveEntry(IndividualMultiObj individual, double noveltyScore) {
            this.individual = individual;
            this.noveltyScore = noveltyScore;
        }

        public IndividualMultiObj getIndividual() {
            return individual;
        }

        public double getNoveltyScore() {
            return noveltyScore;
        }
    }

    public static double[][] generate(List<IndividualMultiObj> individuals, String mode,
                                      double[][] originalObjectives, Map<UUID, Integer> ageInfo,
                                      List<IndividualMultiObj> noveltyArchive, int t, int tMax,
                                      Random random, String currentFilePath) {
        if (individuals.isEmpty()) {
            return new double[0][0];
        }

        double[][] normalizedOriginal = normalizeOriginalObjectives(originalObjectives);

        switch (mode) {
            case MODE_AGE:
                return generateAgeFA(individuals, normalizedOriginal, ageInfo);
            case MODE_GAUSSIAN:
                return generateGaussianFA(individuals, normalizedOriginal, random, currentFilePath);
            case MODE_RECIPROCAL:
                return generateReciprocalFA(individuals, originalObjectives, normalizedOriginal);
            case MODE_PENALTY:
                return generatePenaltyFA(individuals, normalizedOriginal, originalObjectives);
            case MODE_NOVELTY:
                return generateNoveltyFA(individuals, normalizedOriginal, originalObjectives,
                        noveltyArchive.stream()
                                .map(ind -> new ArchiveEntry(ind, calculateNovelty(ind, noveltyArchive, individuals.size()/2)))
                                .collect(Collectors.toList()),
                        individuals.size());
            case MODE_DIVERSITY:
                return generateDiversityFA(individuals, normalizedOriginal, t, tMax);
            default:
                throw new IllegalArgumentException("Unsupported mode: " + mode);
        }
    }

    public static int calculateHammingDistance(Set<Product> products1, Set<Product> products2) {
        if (products1 == null || products2 == null) {
            return Integer.MAX_VALUE;
        }

        int distance = 0;
        int productSize = 0;

        for (Product p1 : products1) {
            for (Product p2 : products2) {
                List<Integer> features1 = new ArrayList<>(p1);
                List<Integer> features2 = new ArrayList<>(p2);

                int minLength = Math.min(features1.size(), features2.size());
                productSize = minLength;

                for (int i = 0; i < minLength; i++) {
                    if (!features1.get(i).equals(features2.get(i))) {
                        distance++;
                    }
                }
            }
        }

        int sizeDiff = Math.abs(products1.size() - products2.size());
        distance += sizeDiff * productSize;

        return distance;
    }

    public static class HammingIndex {
        private final List<IndividualMultiObj> individuals = new ArrayList<>();

        public void add(IndividualMultiObj individual) {
            individuals.add(individual);
        }

        public List<IndividualMultiObj> getNearestNeighbors(IndividualMultiObj query, int k) {
            Set<Product> queryProducts = new HashSet<>(query.getProducts());

            return individuals.stream()
                    .filter(ind -> !ind.equals(query))
                    .map(ind -> {
                        Set<Product> indProducts = new HashSet<>(ind.getProducts());
                        int distance = calculateHammingDistance(queryProducts, indProducts);
                        return new AbstractMap.SimpleEntry<>(ind, distance);
                    })
                    .sorted(Comparator.comparingInt(AbstractMap.SimpleEntry::getValue))
                    .limit(k)
                    .map(AbstractMap.SimpleEntry::getKey)
                    .collect(Collectors.toList());
        }
    }

    public static double calculateNovelty(IndividualMultiObj target,
                                          List<IndividualMultiObj> archive,
                                          int k) {
        if (archive.isEmpty()) {
            return 0.0;
        }

        k = Math.min(k, archive.size());
        k = Math.max(k, 1);

        HammingIndex index = new HammingIndex();
        archive.forEach(index::add);

        List<IndividualMultiObj> neighbors = index.getNearestNeighbors(target, k);
        if (neighbors.isEmpty()) {
            return 0.0;
        }

        double sum = neighbors.stream()
                .mapToDouble(n -> calculateHammingDistance(
                        new HashSet<>(target.getProducts()),
                        new HashSet<>(n.getProducts())))
                .sum();

        return sum / neighbors.size();
    }

    public static void updateNoveltyArchive(List<IndividualMultiObj> population,
                                            List<ArchiveEntry> archive,
                                            int maxSize) {
        int k = Math.max(1, population.size() / 2);

        List<IndividualMultiObj> currentArchive = new ArrayList<>();
        for (ArchiveEntry entry : archive) {
            currentArchive.add(entry.getIndividual());
        }

        List<ArchiveEntry> newEntries = new ArrayList<>();
        for (IndividualMultiObj indiv : population) {
            boolean exists = false;
            for (IndividualMultiObj archived : currentArchive) {
                if (calculateHammingDistance(
                        new HashSet<>(archived.getProducts()),
                        new HashSet<>(indiv.getProducts())) == 0) {
                    exists = true;
                    break;
                }
            }

            if (!exists) {
                double novelty = calculateNovelty(indiv, currentArchive, k);
                newEntries.add(new ArchiveEntry(indiv, novelty));
            }
        }

        archive.addAll(newEntries);

        if (archive.size() > maxSize) {
            List<ArchiveEntry> tempArchive = new ArrayList<>(archive);
            tempArchive.sort(Comparator.comparingDouble(ArchiveEntry::getNoveltyScore));

            List<ArchiveEntry> newArchive = new ArrayList<>(
                    tempArchive.subList(tempArchive.size() - maxSize, tempArchive.size())
            );

            archive.clear();
            archive.addAll(newArchive);
        }
    }

    public static double[][] normalizeOriginalObjectives(double[][] originalObjectives) {
        if (originalObjectives.length == 0) {
            return new double[0][0];
        }

        int numObjectives = originalObjectives[0].length;
        double[] min = new double[numObjectives];
        double[] max = new double[numObjectives];
        Arrays.fill(min, Double.POSITIVE_INFINITY);
        Arrays.fill(max, Double.NEGATIVE_INFINITY);

        for (double[] obj : originalObjectives) {
            for (int i = 0; i < numObjectives; i++) {
                min[i] = Math.min(min[i], obj[i]);
                max[i] = Math.max(max[i], obj[i]);
            }
        }

        for (int i = 0; i < numObjectives; i++) {
            if (min[i] == max[i]) {
                max[i] = min[i] + 1e-9;
            }
        }

        double[][] normalized = new double[originalObjectives.length][numObjectives];
        for (int i = 0; i < originalObjectives.length; i++) {
            for (int j = 0; j < numObjectives; j++) {
                normalized[i][j] = (originalObjectives[i][j] - min[j]) / (max[j] - min[j]);
            }
        }

        return normalized;
    }

    private static double[] normalize(double[] values) {
        if (values == null || values.length == 0) {
            return new double[0];
        }

        double min = Arrays.stream(values).min().orElse(0);
        double max = Arrays.stream(values).max().orElse(0);

        if (min == max) {
            double[] normalized = new double[values.length];
            Arrays.fill(normalized, 0.5);
            return normalized;
        }

        double[] normalized = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            normalized[i] = (values[i] - min) / (max - min);
        }
        return normalized;
    }

    public static double[][] generateAgeFA(List<IndividualMultiObj> individuals,
                                           double[][] normalizedOriginal,
                                           Map<UUID, Integer> ageInfo) {
        int size = individuals.size();
        double[][] faValues = new double[size][2];
        double[] ages = new double[size];

        for (int i = 0; i < size; i++) {
            ages[i] = -ageInfo.getOrDefault(individuals.get(i).getUuid(), 1);
        }

        double[] normalizedAges = normalize(ages);

        for (int i = 0; i < size; i++) {
            faValues[i][0] = normalizedOriginal[i][0] + normalizedAges[i];
            faValues[i][1] = normalizedOriginal[i][0] - normalizedAges[i];
        }

        return faValues;
    }

    public static double[][] generateGaussianFA(List<IndividualMultiObj> individuals,
                                                double[][] normalizedOriginal,
                                                Random random,
                                                String currentFilePath) {
        int size = individuals.size();
        double[][] faValues = new double[size][2];
        double[] ft = Arrays.stream(normalizedOriginal).mapToDouble(arr -> arr[0]).toArray();
        double[] fa = new double[size];

        for (int i = 0; i < size; i++) {
            fa[i] = ft[i] + random.nextGaussian();
        }

        double[] faNormalized = normalize(fa);

        for (int i = 0; i < size; i++) {
            faValues[i][0] = ft[i] + faNormalized[i];
            faValues[i][1] = ft[i] - faNormalized[i];
        }

        return faValues;
    }

    public static double[][] generateReciprocalFA(List<IndividualMultiObj> individuals,
                                                  double[][] originalObjectives,
                                                  double[][] normalizedOriginal) {
        int size = individuals.size();
        double[][] faValues = new double[size][2];
        double[] ft = Arrays.stream(originalObjectives).mapToDouble(arr -> arr[0]).toArray();
        double[] fa = new double[size];

        for (int i = 0; i < size; i++) {
            if (Math.abs(ft[i]) > 1e-9) {
                fa[i] = -1.0 / ft[i];
            } else {
                double lastNonZero = findLastNonZeroFt(ft, i);
                fa[i] = lastNonZero != 0 ? -1.0 / lastNonZero : Double.NEGATIVE_INFINITY;
            }
        }

        double[] faNormalized = normalize(fa);

        for (int i = 0; i < size; i++) {
            faValues[i][0] = normalizedOriginal[i][0] + faNormalized[i];
            faValues[i][1] = normalizedOriginal[i][0] - faNormalized[i];
        }

        return faValues;
    }

    private static double findLastNonZeroFt(double[] ft, int currentIndex) {
        for (int j = currentIndex - 1; j >= 0; j--) {
            if (Math.abs(ft[j]) > 1e-9) {
                return ft[j];
            }
        }
        return 0;
    }

    public static class FeatureValue {
        private final int index;
        private final int value;

        public FeatureValue(int index, int value) {
            this.index = index;
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof FeatureValue)) return false;
            FeatureValue that = (FeatureValue) o;
            return index == that.index && value == that.value;
        }

        @Override
        public int hashCode() {
            return Objects.hash(index, value);
        }
    }

    public static Map<FeatureValue, Double> findPenaltyFeatures(List<IndividualMultiObj> individuals,
                                                                double[][] originalObjectives,
                                                                List<Integer> localOptimaIndices) {
        Map<FeatureValue, Double> penaltyFeatures = new HashMap<>();
        if (localOptimaIndices.isEmpty()) {
            return penaltyFeatures;
        }

        Set<FeatureValue> allFeatures = new HashSet<>();
        for (IndividualMultiObj indiv : individuals) {
            for (Product product : indiv.getProducts()) {
                int position = 0;
                for (Integer featureValue : product) {
                    allFeatures.add(new FeatureValue(position++, featureValue));
                }
            }
        }

        for (FeatureValue feature : allFeatures) {
            double totalFt = 0;
            int count = 0;
            boolean inLocalOptima = false;

            for (int i = 0; i < individuals.size(); i++) {
                IndividualMultiObj indiv = individuals.get(i);
                if (containsFeature(indiv, feature)) {
                    totalFt += originalObjectives[i][0];
                    count++;
                    if (localOptimaIndices.contains(i)) {
                        inLocalOptima = true;
                    }
                }
            }

            if (inLocalOptima && count > 0) {
                double avgFt = totalFt / count;
                double currentPenalty = penaltyFeatures.getOrDefault(feature, 0.0);
                double util_i = avgFt / (1 + currentPenalty);

                if (util_i > 0.3) {
                    penaltyFeatures.put(feature, currentPenalty + util_i);
                }
            }
        }

        return penaltyFeatures;
    }

    private static boolean containsFeature(IndividualMultiObj indiv, FeatureValue feature) {
        for (Product product : indiv.getProducts()) {
            int position = 0;
            for (Integer value : product) {
                if (position == feature.index && value == feature.value) {
                    return true;
                }
                position++;
            }
        }
        return false;
    }

    public static double[][] generatePenaltyFA(List<IndividualMultiObj> individuals,
                                               double[][] normalizedOriginal,
                                               double[][] originalObjectives) {
        List<Integer> localOptimaIndices = findLocalOptima(individuals, originalObjectives);

        Map<FeatureValue, Double> penaltyFeatures = findPenaltyFeatures(
                individuals, normalizedOriginal, localOptimaIndices);

        double[] penalties = new double[individuals.size()];
        for (int i = 0; i < individuals.size(); i++) {
            penalties[i] = calculateIndividualPenalty(individuals.get(i), penaltyFeatures);
        }

        double[] normalizedPenalties = normalize(penalties);

        double[][] faValues = new double[individuals.size()][2];
        for (int i = 0; i < individuals.size(); i++) {
            faValues[i][0] = normalizedOriginal[i][0] + normalizedPenalties[i];
            faValues[i][1] = normalizedOriginal[i][0] - normalizedPenalties[i];
        }

        return faValues;
    }

    private static double calculateIndividualPenalty(IndividualMultiObj indiv,
                                                     Map<FeatureValue, Double> penaltyFeatures) {
        double penalty = 0;
        for (Product product : indiv.getProducts()) {
            int position = 0;
            for (Integer value : product) {
                FeatureValue feature = new FeatureValue(position++, value);
                if (penaltyFeatures.containsKey(feature)) {
                    penalty += penaltyFeatures.get(feature);
                }
            }
        }
        return penalty;
    }

    public static List<Integer> findLocalOptima(List<IndividualMultiObj> individuals,
                                                double[][] originalObjectives
    ) {
        int k=3;
        List<Integer> localOptimaIndices = new ArrayList<>();
        if (individuals.isEmpty() || k <= 0) {
            return localOptimaIndices;
        }

        HammingIndex hammingIndex = new HammingIndex();
        individuals.forEach(hammingIndex::add);

        for (int i = 0; i < individuals.size(); i++) {
            IndividualMultiObj current = individuals.get(i);
            double currentFt = originalObjectives[i][0];
            boolean isLocalOptimum = true;

            List<IndividualMultiObj> neighbors = hammingIndex.getNearestNeighbors(current, k);

            for (IndividualMultiObj neighbor : neighbors) {
                int neighborIndex = individuals.indexOf(neighbor);
                if (originalObjectives[neighborIndex][0] < currentFt) {
                    isLocalOptimum = false;
                    break;
                }
            }

            if (isLocalOptimum) {
                localOptimaIndices.add(i);
            }
        }

        return localOptimaIndices;
    }

    public static double[][] generateDiversityFA(List<IndividualMultiObj> individuals,
                                                 double[][] normalizedOriginal,
                                                 int t, int tMax) {
        int size = individuals.size();
        double[][] faValues = new double[size][2];
        double theta = 1 - ((t - 1.0) / tMax);

        List<Set<Product>> productSets = individuals.stream()
                .map(ind -> new HashSet<>(ind.getProducts()))
                .collect(Collectors.toList());

        double[] rawDiversity = new double[size];
        for (int i = 0; i < size; i++) {
            rawDiversity[i] = calculateFeatureSetDiversity(
                    productSets.get(i),
                    productSets,
                    theta
            );
        }

        double[] normalizedDiversity = normalize(rawDiversity);

        for (int i = 0; i < size; i++) {
            faValues[i][0] = normalizedOriginal[i][0] + normalizedDiversity[i];
            faValues[i][1] = normalizedOriginal[i][0] - normalizedDiversity[i];
        }

        return faValues;
    }

    private static double calculateFeatureSetDiversity(Set<Product> queryProducts,
                                                       List<Set<Product>> populationProducts,
                                                       double theta) {
        List<Integer> distances = new ArrayList<>();

        for (Set<Product> otherProducts : populationProducts) {
            if (!otherProducts.equals(queryProducts)) {
                int distance = calculateHammingDistance(queryProducts, otherProducts);
                distances.add(distance);
            }
        }

        if (distances.isEmpty()) {
            return 0;
        }

        int minNonZeroDist = distances.stream()
                .filter(d -> d > 0)
                .min(Integer::compare)
                .orElse(1);
        double S = theta * minNonZeroDist;
        int S_int = (int) Math.ceil(S);

        long neighborhoodSize = distances.stream()
                .filter(d -> d <= S_int)
                .count();
        int distanceSum = distances.stream()
                .filter(d -> d <= S_int)
                .mapToInt(Integer::intValue)
                .sum();

        return -(neighborhoodSize + distanceSum);
    }

    public static double[][] generateNoveltyFA(List<IndividualMultiObj> individuals,
                                               double[][] normalizedOriginal,
                                               double[][] originalObjectives,
                                               List<ArchiveEntry> noveltyArchive,
                                               int populationSize) {
        int size = individuals.size();
        if (size == 0) {
            return new double[0][0];
        }

        int k = populationSize / 2;
        double[] fa = new double[size];

        for (int i = 0; i < size; i++) {
            IndividualMultiObj indiv = individuals.get(i);
            double novelty = calculateNovelty(indiv,
                    noveltyArchive.stream()
                            .map(ArchiveEntry::getIndividual)
                            .collect(Collectors.toList()),
                    k);
            fa[i] = -novelty;
        }

        double[] faNormalized = normalize(fa);
        double[][] combinedObjectives = new double[size][2];

        for (int i = 0; i < size; i++) {
            combinedObjectives[i][0] = normalizedOriginal[i][0] + faNormalized[i];
            combinedObjectives[i][1] = normalizedOriginal[i][0] - faNormalized[i];
        }

        return combinedObjectives;
    }
}