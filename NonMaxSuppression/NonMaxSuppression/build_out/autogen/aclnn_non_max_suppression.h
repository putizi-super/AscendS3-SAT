
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_NON_MAX_SUPPRESSION_H_
#define ACLNN_NON_MAX_SUPPRESSION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnNonMaxSuppressionGetWorkspaceSize
 * parameters :
 * boxes : required
 * scores : required
 * maxOutputBoxesPerClass : required
 * iouThreshold : required
 * scoreThreshold : required
 * centerPointBoxOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(
    const aclTensor *boxes,
    const aclTensor *scores,
    const aclTensor *maxOutputBoxesPerClass,
    const aclTensor *iouThreshold,
    const aclTensor *scoreThreshold,
    int64_t centerPointBoxOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnNonMaxSuppression
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnNonMaxSuppression(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
