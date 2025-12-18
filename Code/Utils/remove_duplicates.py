def partial_duplicate_replacement(combined_population, population_size):
    fronts = fast_non_dominated_sort(combined_population)

    processed_fronts = []
    for i in range(len(fronts)):
        current_front = fronts[i]

        unique_sols = []
        duplicate_sols = []
        seen_vars = set()

        for sol in current_front:
            var_tuple = tuple(sol.variables)
            if var_tuple not in seen_vars:
                unique_sols.append(sol)
                seen_vars.add(var_tuple)
            else:
                duplicate_sols.append(sol)

        processed_fronts.append(unique_sols)

        if duplicate_sols and i < len(fronts) - 1:
            fronts[i + 1].extend(duplicate_sols)

    new_population = []
    remaining = population_size

    for front in processed_fronts:
        if len(front) <= remaining:
            new_population.extend(front)
            remaining -= len(front)
        else:
            if len(front) > 1:
                crowding_distance_assignment(front)
                front.sort(key=lambda x: -x.attributes['crowding_distance'])

            new_population.extend(front[:remaining])
            remaining = 0

        if remaining == 0:
            break

    return new_population


def fast_non_dominated_sort(population):
    from jmetal.util.ranking import FastNonDominatedRanking
    ranking = FastNonDominatedRanking()
    return ranking.compute_ranking(population)


def crowding_distance_assignment(front):
    from jmetal.util.density_estimator import CrowdingDistance
    density_estimator = CrowdingDistance()
    density_estimator.compute_density_estimator(front)