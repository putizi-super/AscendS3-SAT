
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_REPLICATION_PAD2D_H_
#define ACLNN_REPLICATION_PAD2D_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnReplicationPad2dGetWorkspaceSize
 * parameters :
 * x : required
 * paddings : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReplicationPad2dGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *paddings,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnReplicationPad2d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnReplicationPad2d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
