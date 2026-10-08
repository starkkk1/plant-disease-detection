# AWS readiness — issue #6

Checked 2026-10-08 (Asia/Bangkok). **Deployment is not approved or verified.** Architecture, container recipe and permission review artifacts are prepared; company-specific account readiness is still pending.

## What is known

The user supplied `stark1` as the AWS Console account name. This may be an IAM username or account alias; it is not sufficient to identify an account ID, approved Region, effective IAM role or authenticated CLI profile. No credentials were requested or recorded.

Local checks found no AWS CLI executable and no profile/SSO config at `C:/Users/tranh/.aws/config` or `D:/Code/python/.aws/config`. No authenticated STS/IAM/service calls were made. The Docker client exists but no usable daemon is running. Native API and real-model browser integration work locally. A Console login in a user's browser is not an authenticated connection available to this coding session.

## Company approval record

| Item | Current evidence | Required owner/action |
|---|---|---|
| Console name | `stark1`, supplied by user | Company AWS administrator confirms whether it is a user or account alias |
| AWS account and role | Not verified | Administrator provides an approved SSO/profile or runs the read-only checks below |
| Permitted Region | Not provided | Cloud administrator confirms allowed Region and service availability |
| Approved budget/currency | Not provided | Company cost approver supplies a cap and approval reference |
| Named cost approver | Not identified | Company/project owner names the person authorized to approve cloud spending |
| Resource/operator owner | Not identified | Company assigns ownership, expiry and cleanup responsibility |
| App Runner eligibility | Not verified | Administrator checks whether the account was an existing customer before the new-customer cutoff |
| ECR/App Runner/CloudWatch IAM | Not verified | Review both read permissions and deployment permissions below |
| Optional S3/Cognito/DynamoDB | Not required for MVP | Keep disabled until a specific use and access policy are approved |

The missing owner/approval fields must be supplied by the company; they must not be inferred from the username `stark1`. This checklist is not an IAM approval or a budget authorization.

## Permission review

| Principal | Permission families to verify | Scope/notes |
|---|---|---|
| Read-only auditor | `sts:GetCallerIdentity`, `apprunner:ListServices`, `ecr:DescribeRepositories`, `logs:DescribeLogGroups` | These calls inspect state and do not create resources; success does not prove deployment permission |
| Image publisher | ECR authorization and layer/image push actions | Repository-scoped upload permissions; authorization-token action uses the required broader scope |
| App Runner deployer, if eligible | Create/update/describe/pause/resume service; autoscaling configuration and relevant `iam:PassRole` / service-linked-role permissions | Company reviews resource, Region, tag, SCP, boundary and session-policy constraints |
| App Runner ECR access role | ECR pull permissions; trust principal `build.apprunner.amazonaws.com` | Runtime container does not need an instance role unless future AWS API calls are added |
| ECS Express deployer, if chosen | ECS Express service lifecycle, passing approved execution/infrastructure roles | Administrator verifies the corresponding infrastructure policies and underlying service permissions |
| ECS task execution / Express infrastructure roles | Image pull/log delivery and approved infrastructure provisioning | Separate roles; no developer AdministratorAccess assumption |
| Log reviewer | CloudWatch log/metric read permissions; approved retention policy | Review costs and avoid logging images or credentials |
| Optional application roles | S3 object access, Cognito auth, DynamoDB item permissions only if later added | No optional-service rights required by current classification code |

App Runner role responsibilities are documented by [AWS IAM guidance](https://docs.aws.amazon.com/apprunner/latest/dg/security_iam_service-with-iam.html). AWS notes that App Runner is closed to new customers and recommends an ECS Express alternative, so eligibility must be checked before choosing the service. [Availability change](https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html)

## Read-only checks in an approved CLI/CloudShell session

Replace `APPROVED_REGION` with the company's allowed Region and use only its approved login/profile. The Console name is not automatically a profile name.

```powershell
aws sts get-caller-identity --region APPROVED_REGION
aws apprunner list-services --region APPROVED_REGION --max-results 5
aws ecr describe-repositories --region APPROVED_REGION --max-results 5
aws logs describe-log-groups --region APPROVED_REGION --limit 5
```

These commands were not run here because there is no authenticated CLI. Keep any account/role evidence within the company's permitted record location; never paste access keys, session tokens or credentials into this document. The administrator must separately review effective deployment permissions. A denied list call can mean insufficient read access rather than service unavailability; an allowed list call does not establish create/update rights or new-customer eligibility.

## Review before any paid action

- Record the approved Region, named approver, budget cap and expiry; verify [App Runner Region availability](https://docs.aws.amazon.com/general/latest/gr/apprunner.html) or the chosen ECS Region.
- Use the reviewed permission scope and existing approved roles/repository where possible.
- Build and smoke-test the Linux container on a machine with Docker; verify `/health` and a real `POST /predict`.
- Estimate provisioned memory, active CPU, image storage, logs, traffic and any ECS load balancer/networking charges in the actual Region. App Runner idle provisioned capacity is still billed. [App Runner pricing](https://aws.amazon.com/apprunner/pricing/)
- Cap instance/task count, set log retention, define the person who monitors spend and arrange demo-resource expiry/cleanup.
- Apply the company's required authentication, public access protections and frontend dependency updates before exposing a service publicly.

No paid resources, IAM policies, cloud deployments or AWS credentials were created or changed. The local demo remains the fallback until the company completes the approval and access record.
