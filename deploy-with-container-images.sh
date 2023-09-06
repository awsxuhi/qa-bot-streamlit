#!/bin/bash

# This script deploys the lambda function with the container images.
# -v: show detailed messages when executing command
# -e: exit immediately if any command fail
# set -v 
set -e

# +++++++++++++++++++++++++ Functions +++++++++++++++++++++++++
function log() {
    echo -e "\033[36m[INFO] $1\033[0m"
}

function error_exit() {
    echo -e "\033[31m[ERROR] $1\033[0m" >&2
    exit 1
}

function ensure_repository() {
    local profile=$1
    local region=$2
    local repository_name=$3
    aws --profile $profile ecr describe-repositories --repository-names "$repository_name" --region $region || \
    aws --profile $profile ecr create-repository --repository-name "$repository_name" --region $region
}

function ecr_login() {
    local region=$1
    local account=$2
    local suffix=$3
    local profile=$4
    local repository_name=$5
    local region=$6
    aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.$suffix

    aws --profile $profile ecr set-repository-policy \
    --repository-name "${repository_name}" \
    --policy-text "file://ecr-policy.json" \
    --region ${region}

}

function get_project_name() {
    local config_path="../config/project-config.json"
    python -c "import json; print(json.load(open('$config_path'))['projectName'])"
}

function show_usage() {
    echo "Usage: $0 --name [image-name] --region [region-name] --profile [aws-profile-name]"
    echo "Options:"
    echo "    --name      Name for the Docker image."
    echo "    --region    AWS region name. E.g., ap-northeast-1"
    echo "    --profile   AWS profile name. E.g., default"
    echo "    -h, --help  Show this help message and exit."
}

# +++++++++++++++++++++++++ Main script +++++++++++++++++++++++++
default_profile="default"
default_region=$(aws configure get region --profile $default_profile)
default_repository_name=$(get_project_name)-$(basename $PWD)

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# $#: number of arguments
# $1: the first argument
# $2: the second argument
# $3: the third argument
# shift 2: shift the positional parameters by 2

while [ "$#" -gt 0 ]; do
    case $1 in
        --name)
        repository_name=$2
        shift 2
        ;;
        --region)
        region=$2
        shift 2
        ;;
        --profile)
        profile=$2
        shift 2
        ;;
        *)
        echo "Error: Invalid argument $1"
        show_usage
        exit 1
        ;;
    esac
done

# Check if region and profile are set, if not set default values
repository_name=${repository_name:-$default_repository_name}
region=${region:-$default_region}
profile=${profile:-$default_profile}

suffix="com"
if [[ "$region" == "cn-north-1" ]] || [[ "$region" == "cn-northwest-1" ]]; then
    suffix="com.cn"
fi

account=$(aws sts --profile $profile get-caller-identity --query Account --output text)
if [ $? -ne 0 ]; then
    error_exit "Failed to get the AWS account number."
fi

repository_uri=${account}.dkr.ecr.${region}.amazonaws.$suffix/$repository_name
image_fullname=$repository_uri:latest

log "Ensuring ECR repository exists..."
ensure_repository $profile $region $repository_name

log "Logging in to ECR..."
ecr_login $region $account $suffix $profile $repository_name $region

log "Building Docker image '${repository_name}'..."
docker build --no-cache --platform linux/amd64 -f Dockerfile -t $repository_name  . 

# docker tag qabot-question-answering:latest 581725073534.dkr.ecr.ap-northeast-1.amazonaws.com/qabot-question-answering:latest
log "Tagging Docker image..."
docker tag "$repository_name:latest" $image_fullname

log "Pushing Docker image to ECR..."
docker push $image_fullname

echo -e "\033[32mDone.\033[0m"
