# How to join the community

# How the Community Works

## Roadmap

We update the community Roadmap periodically, usually on a quarterly basis. New projects added within the quarter will also be synchronized into the Roadmap. The Roadmap for each cycle will be presented in the form of an issue and pinned to the top of the Issues list, where you can see what the community currently aims to achieve.

Generally, we default to using the content in the Roadmap more for recording and displaying rather than as a time-bound commitment. We hope that community work is continuous, healthy, and can be joined at any time. Projects usually do not need to be bound by a specific deadline, which could lead to the inability to produce truly valuable and recognized work. **Create momentum – don't sprint**.

Example: https://github.com/sgl-project/sglang-jax/issues/190

## Project

Every piece of work with a relatively long cycle (greater than two weeks) or evaluated as having a significant impact (risk of blocking other parallel development projects) needs to exist in the form of a project. Each project will be placed in the current cycle's roadmap before it ends; if not completed, it will be postponed to the next roadmap.

Each project will be associated with an issue, which is used to record the project's description, associated design documents, ways to join (usually a project-level slack/discord channel), and relevant participant information. When the project itself is in the development stage, finer-grained functions will continue to be broken down into sub-issues.

## Issues

Each specific Issue undertakes specific development content and is associated with a Pull Request. Usually, when an Issue is not assigned to anyone, it is defaulted to a state where development is planned but is waiting for interested developers to join.

## Linear

We use Linear (https://linear.app/) to manage and track the progress of all projects and sync them to Github, and it is used for collaboration with multiple open-source projects. We plan to open up the entire Linear in the future and synchronize more complete project details and progress information to the open-source community.

# How You Can Contribute

## Try Joining or Initiating a Project

You can find all ongoing projects in the current roadmap, click on the issue associated with the project to view the current status. Usually, you can open the design doc recorded in the project issue to view the current project's design plan, and join discord to communicate directly with project members to ask about current progress.

If you have a topic of interest that is not in the roadmap, you are very welcome to directly open an issue, and we will contact you as soon as possible to discuss how to implement it!

## Try Implementing a Model

The goal of the SGLang-Jax project itself is a high-performance inference engine. The superposition of various functions and optimizations makes even a simple model implementation very complex. Considering this issue, we divide model implementation into two types: unimplemented models and unoptimized models.

For unimplemented models, we collaborate with the Bonsai project (https://github.com/jax-ml/bonsai). We can directly open an issue under that project and communicate implementation details and work in Discord. We hope this work serves as a minimally optimized, correct model implementation, used as a base model for various future optimization explorations.

For unoptimized models, we will conduct a series of optimization works based on the base model. While ensuring alignment in performance and correctness values, we add new optimization features. This work often exists in the form of a project.

## Try Contributing via Test Cases and Documentation

You can contribute to SGLang-Jax by adding test cases or modifying documentation as an entry-level way to contribute.

At the same time, if you are unfamiliar with the project in the early stages, it is also recommended to use AI tools to help you choose exactly which test cases or documents to write. For example, you can use Deep Research capabilities and appropriate prompts at https://deepwiki.com/sgl-project/sglang-jax to find documents and tests that can be optimized. Similarly, we have also applied to the https://codewiki.google/ project. We strongly encourage the reasonable use of AI tools to deepen your understanding of the project and make some contributions!

# Recommended Reading

* https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/jax_tutorial.md
* https://github.com/sgl-project/sglang-jax/blob/main/docs/developer_guide/contribution_guide.md
