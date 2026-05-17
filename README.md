## Modifications from the Original End2Race

This project is a modified version of the original [End2Race](https://github.com/michigan-traffic-lab/End2Race), created to address issues encountered while adapting the original project to our experimental setup. While the original project mainly demonstrates high-speed racing scenarios, this version focuses on problems that arise when using the framework in low-speed driving conditions.

The original imitation learning pipeline and Lattice Planner-based expert demonstration structure are preserved. However, the collision cost computation was modified for multi-agent overtaking scenarios.

In the original implementation, collision risk was evaluated based on the closest point between the opponent vehicle and each candidate trajectory. However, in low-speed driving, even if the ego vehicle selects the same trajectory, the actual encounter timing with the opponent can vary depending on the selected speed candidate. Therefore, a purely distance-based collision check may not accurately reflect the collision risk.

In this modified version, the opponent vehicle’s velocity is estimated using its previous and current positions. Then, the expected position of the opponent is predicted according to the time at which the ego vehicle reaches each point on the candidate trajectory. As a result, collision risk is evaluated differently depending on the selected trajectory-speed candidate, enabling a more time-aware collision-aware trajectory selection process.

Additionally, a debug option was added to selectively print internal planner cost components, making it easier to analyze why a specific trajectory was selected.
